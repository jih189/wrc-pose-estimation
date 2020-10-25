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
from chamfer2D.dist_chamfer_2D import chamfer_2DDist
from poseUtil import getPredictPose, getRotationError, ADD_error, ADDS_error

import torchgeometry as tgm
import kornia

import src.configuration as CFG

import cv2

# import wandb

# wandb.init(project="wrc-pose-refinement")

batch_size = 64
epochs = 150
lr = 4e-5
momentum = 0.9
w_decay = 0.1
seglambda = 0.5
flowlambda = 5.0
train_dir = CFG.REFINE_DATA_PATH
val_dir = CFG.REFINE_DATA_PATH

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
mymodel = DeepIM().cuda()
mymodel = nn.DataParallel(mymodel)
mymodel.module.flownet.load_state_dict(
    torch.load(CFG.BEST_MODEL_FLOWNET).module.state_dict()
)
mymodel.module.flownet.eval()


# validation setup
# mymodel = torch.load(CFG.BEST_MODEL_REFINE)
# mymodel.eval()
# wandb.watch(mymodel)

seg_criterion = nn.CrossEntropyLoss(reduce=False)
cham_criterion = chamfer_2DDist()

optimizer = optim.Adam(
    mymodel.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-08, weight_decay=w_decay
)

lmbda = lambda epoch: 0.5
scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

torch.autograd.set_detect_anomaly(True)


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

            origin_pt = torch.tensor([0, 0, 0]).view(1, 1, 3).cuda().float()
            origin_pt = origin_pt.repeat(init3dPt.shape[0], init3dPt.shape[1], 1)
            origin_pt = tgm.transform_points(targetPose, origin_pt)

            # get distance between each point pairs in 3d
            targetFromInit3dPt = tgm.transform_points(targetPose, init3dPt)
            predictVec = predict3dpts - origin_pt
            targetVec = targetFromInit3dPt - origin_pt
            cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
            cosineSim = cos(predictVec, targetVec)
            cosloss = -torch.mean(cosineSim, 1).sum()

            # get normal distance
            distanceBetweenVec3d = predict3dpts - targetFromInit3dPt
            dist3dloss = torch.mean(
                torch.norm(distanceBetweenVec3d, p=1, dim=2), 1
            ).sum()

            loss = cosloss + flowloss + segloss
            # loss = dist3dloss + flowloss + segloss
            loss.backward()
            optimizer.step()
            avg_loss.append(loss.data.cpu().numpy().sum())

        tem = sum(avg_loss) / len(train_dataset)
        print("Finish epoch {}, loss {}".format(epoch, tem))

        val_loss = val()
        mymodel.train()

        if pre_loss == None:
            torch.save(mymodel, CFG.BEST_MODEL_REFINE)
            pre_loss = val_loss
        elif pre_loss < val_loss:
            torch.save(mymodel, CFG.BEST_MODEL_REFINE)
            pre_loss = val_loss
        mymodel.train()
        if (epoch + 1) % 30 == 0:
            scheduler.step()


def val():
    mymodel.eval()
    avg_loss = []
    avg_rot_error = []
    avg_trans_error = []
    avg_add_match_rate = []
    avg_adds_match_rate = []
    avg_iou = []

    avg_dist3d_loss = []
    avg_flow_loss = []
    avg_seg_loss = []
    avg_chamfer_loss = []
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

        # get chamfer distance between predict and label
        predict_2d_pts = kornia.project_points(predict3dpts, camera_matrix)
        target_2d_pts = kornia.project_points(targetFromInit3dPt, camera_matrix)
        dist1, dist2, idx1, idx2 = cham_criterion(
            predict_2d_pts.float(), target_2d_pts.float()
        )
        ch_loss = torch.mean(dist1, 1) + torch.mean(dist2, 1)
        ch_loss = ch_loss.sum()

        # rotation error
        rotError = getRotationError(pred_pose, targetPose)
        # rotError = getRotationError(initPose, targetPose)

        # translation error
        transError = torch.norm(pred_pose[:, :3, 3] - targetPose[:, :3, 3], p=2, dim=1)

        # ADD rate
        addMatchRate = ADD_error(pred_pose, targetPose)
        # ADDS rate
        addsMatchRate = ADDS_error(pred_pose, targetPose)
        # print(addsMatchRate[:3])
        # get normal distance
        dist3dloss = torch.mean(torch.norm(distanceBetweenVec3d, p=1, dim=2), 1).sum()

        origin_pt = torch.tensor([0, 0, 0]).view(1, 1, 3).cuda().float()
        origin_pt = origin_pt.repeat(init3dPt.shape[0], init3dPt.shape[1], 1)
        origin_pt = tgm.transform_points(targetPose, origin_pt)

        # get distance between each point pairs in 3d
        predictVec = predict3dpts - origin_pt
        targetVec = targetFromInit3dPt - origin_pt
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        cosineSim = cos(predictVec, targetVec)
        cosloss = -torch.mean(cosineSim, 1).sum()

        # loss = dist3dloss + flowloss + segloss
        loss = cosloss + flowloss + segloss

        iou_value = iou(labelmask_img, segmentMask, 1)

        avg_loss.append(loss.data.cpu().numpy().sum())
        avg_rot_error.append(rotError.sum())
        avg_add_match_rate.append(addMatchRate.sum())
        avg_adds_match_rate.append(addsMatchRate.sum())
        avg_trans_error.append(transError.data.cpu().numpy().sum())
        avg_iou.append(iou_value.data.cpu().numpy())

        avg_dist3d_loss.append(dist3dloss.data.cpu().numpy().sum())
        avg_flow_loss.append(flowloss.data.cpu().numpy())
        avg_seg_loss.append(segloss.data.cpu().numpy())
        avg_chamfer_loss.append(ch_loss.data.cpu().numpy())

    tem = sum(avg_loss) / len(val_dataset)
    tem_rot_error = sum(avg_rot_error) / len(val_dataset)
    term_add_rate = sum(avg_add_match_rate).float() / len(val_dataset)
    term_adds_rate = sum(avg_adds_match_rate).float() / len(val_dataset)
    tem_trans_error = sum(avg_trans_error) / len(val_dataset)
    ioutem = sum(avg_iou) / len(val_dataset)
    dist3dlosstem = sum(avg_dist3d_loss) / len(val_dataset)
    flowlosstem = sum(avg_flow_loss) / len(val_dataset)
    seglosstem = sum(avg_seg_loss) / len(val_dataset)
    chamferlosstem = sum(avg_chamfer_loss) / len(val_dataset)
    print(
        "val loss {}, rot_error {}, trans_error {} iou {} chamfer_loss {} add rate {}% adds rate {}%".format(
            tem,
            tem_rot_error,
            tem_trans_error,
            ioutem,
            chamferlosstem,
            term_add_rate * 100,
            term_adds_rate * 100,
        )
    )
    print(
        "distance 3d loss {} flow loss {} seg loss {}".format(
            dist3dlosstem, flowlosstem, seglosstem
        )
    )
    return term_add_rate


if __name__ == "__main__":
    # val()
    train()
    # print("done")
    # initPose rot error = 0.35

