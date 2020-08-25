# import chamfer2D.dist_chamfer_2D as CHAMFER2D
from src.common.DataLoader import Refine_data
from models.model import DeepIM, FlowNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import random
import numpy as np
from src.common.iou import iou

import torchgeometry as tgm
import kornia
import src.configuration as CFG
from scipy.spatial.transform import Rotation as R

import cv2

# import wandb

# wandb.init(project="wrc-pose-refinement")

##########init PSPNet #####################
# PSPmodel = PSPNet().cuda()
# PSPmodel = nn.DataParallel(PSPmodel)
# PSPmodel = torch.load("best_model_psp.pth")
# PSPmodel.eval()
###########################################


##########init flownet #####################
# flow_model = FlowNet().cuda()

# flow_model = nn.DataParallel(flow_model)
# flow_model = torch.load("best_model_flownet.pth")
# flow_model.eval()
###########################################

batch_size = 64
epochs = 1000
lr = 1e-5
momentum = 0.9
w_decay = 0.1
seglambda = 10.0
flowlambda = 1.0

train_dir = CFG.REFINE_SATA_PATH
val_dir = CFG.REFINE_SATA_PATH

# build train data loader
train_dataset = Refine_data(data_path=train_dir, isTrain=True)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=16
)

# build val data loader
val_dataset = Refine_data(data_path=val_dir, isTrain=False)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=16
)

# initiate the net
# mymodel = Refine_Net().cuda()
mymodel = DeepIM().cuda()
mymodel = nn.DataParallel(mymodel)
mymodel.module.flownet.load_state_dict(
    torch.load("best_model_flownet.pth").module.state_dict()
)
mymodel.module.flownet.eval()

# wandb.watch(mymodel)

seg_criterion = nn.CrossEntropyLoss(reduce=False)

# optimizer = optim.SGD(
#     list(mymodel.module.fc1.parameters()) + list(mymodel.module.fc2.parameters()) + list(mymodel.module.fcrotation.parameters())+
#     list(mymodel.module.fctraslation.parameters()) + list(mymodel.module.fcdist.parameters())
#     , lr=lr, momentum=momentum, weight_decay=w_decay
# )

# optimizer = optim.SGD(
#     mymodel.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay
# )

optimizer = optim.Adam(
    mymodel.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-08, weight_decay=w_decay
)

lmbda = lambda epoch: 0.5
scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

# negative pair
NEG_ONE_PAIR = torch.tensor([[-1.0, -1.0]]).cuda()

torch.autograd.set_detect_anomaly(True)


def getPredictPose(initPose, rot, trans, dist, imagesize, rescaleValue):
    # convert the shift to right format
    trans = (trans - 0.5) * imagesize

    initPoseRot = torch.eye(4).repeat(trans.shape[0], 1, 1).cuda().float()
    initPoseRot[:, :3, :3] = initPose[:, :3, :3]

    # generate the rotation pose
    rot_pose = torch.bmm(tgm.angle_axis_to_rotation_matrix(rot), initPoseRot)
    rot_pose[:, :3, 3] = initPose[:, :3, 3]

    # apply the predicted trans and dist to the rot pose, so we can get
    # the predict pose
    dist_pose = rot_pose.clone()
    dist_pose[:, 2, 3] = rot_pose[:, 2, 3] / dist[:, 0, 0]

    horizontalR = torch.atan2(
        trans[:, :, 0].view(trans.shape[0]),
        torch.tensor(CFG.CAMERA_MATRIX[0, 0]).cuda() * rescaleValue,
    )

    verticalR = -torch.atan2(
        trans[:, :, 1].view(trans.shape[0]),
        torch.sqrt(
            trans[:, :, 0].view(trans.shape[0]) * trans[:, :, 0].view(trans.shape[0])
            + torch.tensor(CFG.CAMERA_MATRIX[1, 1] * CFG.CAMERA_MATRIX[1, 1]).cuda()
            * rescaleValue
            * rescaleValue
        )
        * torch.tensor(CFG.CAMERA_MATRIX[1, 1] / CFG.CAMERA_MATRIX[1, 1]).cuda(),
    )

    ch = torch.cos(horizontalR)
    sh = torch.sin(horizontalR)
    cb = torch.cos(verticalR)
    sb = torch.sin(verticalR)
    ca = torch.cos(torch.zeros(trans.shape[0]).cuda())  # z axiz
    sa = torch.sin(torch.zeros(trans.shape[0]).cuda())  # z axiz

    m00 = ch * ca
    m01 = sh * sb - ch * sa * cb
    m02 = ch * sa * sb + sh * cb
    m10 = sa
    m11 = ca * cb
    m12 = -ca * sb
    m20 = -sh * ca
    m21 = sh * sa * cb + ch * sb
    m22 = -sh * sa * sb + ch * cb

    rotation_matrix = torch.eye(4).repeat(trans.shape[0], 1, 1).cuda()
    m00 = m00.unsqueeze(0)
    m01 = m01.unsqueeze(0)
    m02 = m02.unsqueeze(0)
    m10 = m10.unsqueeze(0)
    m11 = m11.unsqueeze(0)
    m12 = m12.unsqueeze(0)
    m20 = m20.unsqueeze(0)
    m21 = m21.unsqueeze(0)
    m22 = m22.unsqueeze(0)

    m = torch.cat((m00, m01, m02, m10, m11, m12, m20, m21, m22), dim=0)
    m.transpose_(0, 1)
    rotation_matrix[..., :3, :3] = m.view(-1, 3, 3)

    # get predicted pose
    pred_pose = torch.bmm(rotation_matrix, dist_pose)

    return pred_pose


def getRotationError(pred_pose, targetPose):
    pred_rot = pred_pose[:, :3, :3]
    pred_rot_T = torch.transpose(pred_rot, 1, 2)
    target_rot = targetPose[:, :3, :3]
    rot_delta = torch.bmm(target_rot, pred_rot_T)
    rot_delta = rot_delta.data.cpu().numpy()
    rot_delta_angle_axis = np.array(
        [R.from_matrix(rot_mtx).as_rotvec() for rot_mtx in rot_delta]
    )
    rot_delta_angle = np.linalg.norm(rot_delta_angle_axis, axis=1)
    return rot_delta_angle


def train():
    print("start training...")
    pre_loss = None
    for epoch in range(epochs):
        avg_loss = []
        for data in train_loader:
            (
                idx,
                input_img,
                edge_img,
                mask_img,
                init3dPt,
                initPose,
                target3dPt,
                targetPose,
                rescaleValue,
                labelmask_img,
                flow_img,
            ) = data

            # load data to cuda
            input_img = input_img.cuda().float() / 255.0
            edge_img = edge_img.cuda().float() / 255.0
            mask_img = mask_img.cuda().float() / 255.0
            init3dPt = init3dPt.cuda().float()
            initPose = initPose.cuda().float()
            target3dPt = target3dPt.cuda().float()
            targetPose = targetPose.cuda().float()
            rescaleValue = rescaleValue.cuda().float()
            labelflow = flow_img[:, :2, :, :].cuda().float()

            labelmask_img = Variable(labelmask_img).cuda().long()
            labelmask_img = labelmask_img.squeeze(1)

            flow_inputData = torch.cat((mask_img, edge_img, input_img), 1,)
            flow_input = Variable(flow_inputData)

            optimizer.zero_grad()

            ################ test
            # testindex = 0
            # testimg = np.transpose(mask_img[testindex].cpu().detach().numpy(), (1, 2, 0)).copy()
            ################

            # predict the rotation, translation, and depth in image view
            rot, trans, dist, opticalFlow, segmentMask = mymodel(flow_input)

            opticalFlow = torch.sigmoid(opticalFlow)

            segmentMask = segmentMask.squeeze(1)

            segloss = seg_criterion(segmentMask, labelmask_img)
            segloss = torch.mean(segloss.view(segloss.shape[0], -1), dim=1)
            segloss = segloss.sum() * seglambda

            maskarea = torch.sum(mask_img.view(mask_img.shape[0], -1), dim=1)

            flowloss = torch.norm(opticalFlow - labelflow, p=1, dim=1)
            flowloss = flowloss.unsqueeze(1)
            flowloss = flowloss * mask_img
            flowloss = torch.sum(flowloss.view(flowloss.shape[0], -1), dim=1)
            flowloss = flowloss / maskarea

            flowloss = flowloss.sum() * flowlambda

            trans = trans.unsqueeze(1)
            dist = dist.unsqueeze(1)

            # update the camera matrix because the input image is resize
            camera_matrix_original = (
                torch.tensor(
                    [
                        [
                            [
                                CFG.CAMERA_MATRIX[0, 0],
                                0.0,
                                float(input_img.shape[-2]) / 2,
                            ],
                            [
                                0.0,
                                CFG.CAMERA_MATRIX[1, 1],
                                float(input_img.shape[-2]) / 2,
                            ],
                            [0.0, 0.0, 1.0],
                        ]
                    ]
                )
                .repeat(trans.shape[0], 1, 1)
                .cuda()
            )
            camera_matrix = camera_matrix_original.clone()
            camera_matrix[:, 0, 0] = camera_matrix_original[:, 0, 0] * rescaleValue
            camera_matrix[:, 1, 1] = camera_matrix_original[:, 1, 1] * rescaleValue
            camera_matrix.unsqueeze_(1)

            pred_pose = getPredictPose(
                initPose, rot, trans, dist, input_img.shape[-2], rescaleValue
            )

            # predict 3d pts
            predict3dpts = tgm.transform_points(pred_pose, init3dPt)

            ######################################################## test predict 2d pts
            # for p in predict_2d_pts.cpu().detach().numpy()[testindex]:
            #     testimg = cv2.circle(
            #         testimg,
            #         (int(p[0]), int(p[1])),
            #         radius=0,
            #         color=(255, 0, 0),
            #         thickness=-1,
            #     )
            # cv2.imshow("test", testimg)
            # cv2.waitKey(0)
            #############################################################

            # wandb.log

            # get distance between each point pairs in 3d
            targetFromInit3dPt = tgm.transform_points(targetPose, init3dPt)

            # get normal distance
            distanceBetweenVec3d = predict3dpts - targetFromInit3dPt
            dist3dloss = torch.mean(
                torch.norm(distanceBetweenVec3d, p=1, dim=2), 1
            ).sum()

            loss = dist3dloss + flowloss + segloss
            loss.backward()
            optimizer.step()
            avg_loss.append(loss.data.cpu().numpy().sum())

        tem = sum(avg_loss) / len(train_dataset)
        print("Finish epoch {}, loss {}".format(epoch, tem))

        val_loss = val()
        mymodel.train()

        # if pre_loss == None:
        #     torch.save(mymodel, "best_model_refine_housing.pth")
        #     pre_loss = val_loss
        # elif pre_loss > val_loss:
        #     torch.save(mymodel, "best_model_refine_housing.pth")
        #     pre_loss = val_loss
        # mymodel.train()
        # if (epoch + 1) % 50 == 0:
        #     scheduler.step()


def val():
    mymodel.eval()
    avg_loss = []
    avg_rot_error = []
    avg_trans_error = []
    avg_iou = []

    avg_dist3d_loss = []
    avg_flow_loss = []
    avg_seg_loss = []
    for data in val_loader:
        (
            idx,
            input_img,
            edge_img,
            mask_img,
            init3dPt,
            initPose,
            target3dPt,
            targetPose,
            rescaleValue,
            labelmask_img,
            flow_img,
        ) = data

        input_img = input_img.cuda().float() / 255.0
        edge_img = edge_img.cuda().float() / 255.0
        mask_img = mask_img.cuda().float() / 255.0
        init3dPt = init3dPt.cuda().float()
        initPose = initPose.cuda().float()
        target3dPt = target3dPt.cuda().float()
        targetPose = targetPose.cuda().float()
        rescaleValue = rescaleValue.cuda().float()
        labelflow = flow_img[:, :2, :, :].cuda().float()

        labelmask_img = Variable(labelmask_img).cuda().long()
        labelmask_img = labelmask_img.squeeze(1)

        flow_inputData = torch.cat((mask_img, edge_img, input_img), 1,)

        flow_input = Variable(flow_inputData)

        with torch.no_grad():
            rot, trans, dist, opticalFlow, segmentMask = mymodel(flow_input)

        opticalFlow = torch.sigmoid(opticalFlow)

        segmentMask = segmentMask.squeeze(1)

        segloss = seg_criterion(segmentMask, labelmask_img)
        segloss = torch.mean(segloss.view(segloss.shape[0], -1), dim=1)
        segloss = segloss.sum() * seglambda

        maskarea = torch.sum(mask_img.view(mask_img.shape[0], -1), dim=1)

        flowloss = torch.norm(opticalFlow - labelflow, p=1, dim=1)
        flowloss = flowloss.unsqueeze(1)
        flowloss = flowloss * mask_img
        flowloss = torch.sum(flowloss.view(flowloss.shape[0], -1), dim=1)
        flowloss = flowloss / maskarea

        flowloss = flowloss.sum() * flowlambda

        trans = trans.unsqueeze(1)
        dist = dist.unsqueeze(1)

        # update the camera matrix because the input image is resize
        camera_matrix_original = (
            torch.tensor(
                [
                    [
                        [CFG.CAMERA_MATRIX[0, 0], 0.0, float(input_img.shape[-2]) / 2,],
                        [0.0, CFG.CAMERA_MATRIX[1, 1], float(input_img.shape[-2]) / 2,],
                        [0.0, 0.0, 1.0],
                    ]
                ]
            )
            .repeat(trans.shape[0], 1, 1)
            .cuda()
        )
        camera_matrix = camera_matrix_original.clone()
        camera_matrix[:, 0, 0] = camera_matrix_original[:, 0, 0] * rescaleValue
        camera_matrix[:, 1, 1] = camera_matrix_original[:, 1, 1] * rescaleValue
        camera_matrix.unsqueeze_(1)

        pred_pose = getPredictPose(
            initPose, rot, trans, dist, input_img.shape[-2], rescaleValue
        )

        # predict 3d pts
        predict3dpts = tgm.transform_points(pred_pose, init3dPt)

        # get distance between each point pairs in 3d
        targetFromInit3dPt = tgm.transform_points(targetPose, init3dPt)
        distanceBetweenVec3d = predict3dpts - targetFromInit3dPt

        # rotation error
        rotError = getRotationError(pred_pose, targetPose)

        # translation error
        transError = torch.norm(pred_pose[:, :3, 3] - targetPose[:, :3, 3], p=2, dim=1)

        # get normal distance
        dist3dloss = torch.mean(torch.norm(distanceBetweenVec3d, p=1, dim=2), 1).sum()

        loss = dist3dloss + flowloss + segloss

        iou_value = iou(labelmask_img, segmentMask, 1)

        avg_loss.append(loss.data.cpu().numpy().sum())
        avg_rot_error.append(rotError.sum())
        avg_trans_error.append(transError.data.cpu().numpy().sum())
        avg_iou.append(iou_value.data.cpu().numpy())

        avg_dist3d_loss.append(dist3dloss.data.cpu().numpy().sum())
        avg_flow_loss.append(flowloss.data.cpu().numpy())
        avg_seg_loss.append(segloss.data.cpu().numpy())

    tem = sum(avg_loss) / len(val_dataset)
    tem_rot_error = sum(avg_rot_error) / len(val_dataset)
    tem_trans_error = sum(avg_trans_error) / len(val_dataset)
    ioutem = sum(avg_iou) / len(val_dataset)
    dist3dlosstem = sum(avg_dist3d_loss) / len(val_dataset)
    flowlosstem = sum(avg_flow_loss) / len(val_dataset)
    seglosstem = sum(avg_seg_loss) / len(val_dataset)
    print(
        "val loss {}, rot_error {}, trans_error {} iou {}".format(
            tem, tem_rot_error, tem_trans_error, ioutem
        )
    )
    print(
        "distance 3d loss {} flow loss {} seg loss {}".format(
            dist3dlosstem, flowlosstem, seglosstem
        )
    )
    return tem


if __name__ == "__main__":
    # val()
    train()
    print("done")
    # initPose rot error = 0.35

