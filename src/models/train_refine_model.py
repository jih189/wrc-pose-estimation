import chamfer2D.dist_chamfer_2D as CHAMFER2D
from src.common.DataLoader import Refine_data
from models.model import Refine_Net, PSPNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import random

import torchgeometry as tgm
import kornia
import src.common.fscore as FS
import src.configuration as CFG

# import wandb

# wandb.init(project="wrc-pose-refinement")

##########init PSPNet #####################
PSPmodel = PSPNet().cuda()
PSPmodel = nn.DataParallel(PSPmodel)
PSPmodel = torch.load("best_model_psp.pth")
PSPmodel.eval()
###########################################

LOG_INTERVAL = 1

batch_size = 32
epochs = 1200
lr = 1e-6
momentum = 0.9
w_decay = 18.0
lambda_chamfer = 0.05
lambda3d = 100.0

train_dir = "data/processed/pulley_refine/"
val_dir = "data/processed/pulley_refine/"

num_images = 5773
train_ratio = 0.8
num_train = int(train_ratio * num_images)
num_val = num_images - num_train

# use for spliting train and test
val_list = random.sample(range(num_images), num_val)
train_list = [i for i in range(num_images) if i not in val_list]

f = open(train_dir + "train.txt", "w")
for i in train_list:
    f.write("{:06d}".format(i) + "\n")
f.close()

f = open(val_dir + "val.txt", "w")
for i in val_list:
    f.write("{:06d}".format(i) + "\n")
f.close()
print("Split training and validation set done")

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
mymodel = Refine_Net().cuda()
mymodel = nn.DataParallel(mymodel)

# wandb.watch(mymodel)

# loss for chamfer
chamLoss = CHAMFER2D.chamfer_2DDist()

optimizer = optim.SGD(
    mymodel.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay
)

# optimizer = optim.Adam(
#     mymodel.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-08, weight_decay=w_decay
# )

lmbda = lambda epoch: 0.5
scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

# negative pair
NEG_ONE_PAIR = torch.tensor([[-1.0, -1.0]]).cuda()

torch.autograd.set_detect_anomaly(True)

softmax = nn.Softmax2d()


def train():
    print("start training... Nahid habibi")
    pre_loss = None
    for epoch in range(epochs):
        avg_loss = []
        avg_f_score = []
        avg_chamfer_score = []
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
            ) = data

            predictMask = PSPmodel(input_img.cuda().float())


            predictMask = torch.argmax(predictMask, 1, keepdim=True)
            predictMask = (predictMask == 1).detach()

            optimizer.zero_grad()

            # catenate image, edges, mask, and bounding box into one input
            # input order (edge, mask, bounding box, image)
            inputData = torch.cat(
                (edge_img.cuda().float(), mask_img.cuda().float(), predictMask.float(), input_img.cuda().float()), 1
            )
            input = Variable(inputData)

            ################ test
            # testindex = 2
            # testimg = np.transpose(input_img[testindex].numpy(), (1, 2, 0)).copy()
            ################

            # load data to cuda
            init3dPt = init3dPt.cuda().float()
            target3dPt = target3dPt.cuda().float()
            initPose = initPose.cuda().float()
            targetPose = targetPose.cuda().float()
            rescaleValue = rescaleValue.cuda().float()

            # predict the rotation, translation, and depth in image view
            rot, trans, dist = mymodel(input)

            trans = trans.unsqueeze(1)
            dist = dist.unsqueeze(1)

            # convert the shift to right format
            trans = (trans - 0.5) * input_img.shape[-1]

            # generate the rotation pose
            rot_pose = torch.bmm(initPose, tgm.angle_axis_to_rotation_matrix(rot))

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
                                float(input_img.shape[-1]) / 2,
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

            # get the target 2d points
            target3dPt = tgm.transform_points(targetPose, target3dPt)
            target_2d_pts = kornia.project_points(target3dPt, camera_matrix)

            # get the predict 2d points
            predict3dpt = tgm.transform_points(rot_pose, init3dPt)
            predict_2d_pts = kornia.project_points(predict3dpt, camera_matrix)

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
                    trans[:, :, 0].view(trans.shape[0])
                    * trans[:, :, 0].view(trans.shape[0])
                    + torch.tensor(
                        CFG.CAMERA_MATRIX[1, 1] * CFG.CAMERA_MATRIX[1, 1]
                    ).cuda()
                    * rescaleValue
                    * rescaleValue
                )
                * torch.tensor(
                    CFG.CAMERA_MATRIX[1, 1] / CFG.CAMERA_MATRIX[1, 1]
                ).cuda(),
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

            # predict 3d pts
            predict3dpts = tgm.transform_points(pred_pose, init3dPt)

            ############################################# test
            # test_2d_pts = kornia.project_points(predict3dpts, camera_matrix)
            # for p in test_2d_pts.cpu().detach().numpy()[testindex]:
            #     testimg = cv2.circle(
            #         testimg,
            #         (int(p[0]), int(p[1])),
            #         radius=0,
            #         color=(0, 255, 0),
            #         thickness=-1,
            #     )
            # cv2.imshow("test", testimg)
            # cv2.waitKey(0)
            #################################################

            # apply the translation on 2d points
            center = torch.tensor(
                [[float(input_img.shape[-2]) / 2, float(input_img.shape[-1]) / 2]]
            ).cuda()

            # scale the 2d points with dist
            scaled_predict_2d_pts = torch.add(
                torch.add(predict_2d_pts, center * NEG_ONE_PAIR) * dist, center
            )

            # shift the 2d points with trans
            predict_2d_pts = torch.add(scaled_predict_2d_pts, trans)

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

            # calculate the chamfer distance loss
            dist1, dist2, idx1, idx2 = chamLoss(predict_2d_pts.float(), target_2d_pts.float())

            # get distance between each point pairs in 3d
            targetFromInit3dPt = tgm.transform_points(targetPose, init3dPt)
            distanceBetweenVec3d = predict3dpts - targetFromInit3dPt

            # get distance between each point pairs in 2d
            target_2d_pts_init = kornia.project_points(
                targetFromInit3dPt, camera_matrix
            )

            ######################################## show target pose
            # for p in target_2d_pts_init.cpu().detach().numpy()[testindex]:
            #     testimg = cv2.circle(
            #         testimg,
            #         (int(p[0]), int(p[1])),
            #         radius=0,
            #         color=(0, 0, 255),
            #         thickness=-1,
            #     )
            # cv2.imshow("test", testimg)
            # cv2.waitKey(0)
            ##########################################

            distanceBetweenVec = predict_2d_pts - target_2d_pts_init

            # get normal distance
            normdist2d = torch.norm(distanceBetweenVec, p=1, dim=2)
            normdist3d = torch.norm(distanceBetweenVec3d, p=2, dim=2)

            ch_loss = torch.mean(dist1, 1) + torch.mean(dist2, 1)
            loss = torch.mean(normdist3d, 1) * lambda3d + torch.mean(normdist2d, 1)

            loss.sum().backward()
            optimizer.step()
            avg_loss.append(loss.data.cpu().numpy().sum())
            f_score, precision, recall = FS.fscore(dist1, dist2)
            avg_f_score.append(f_score.data.cpu().numpy().sum())
            avg_chamfer_score.append(ch_loss.data.cpu().numpy().sum())

        tem = sum(avg_loss) / len(train_dataset)
        tem_f_score = sum(avg_f_score) / len(train_dataset)
        tem_chamfer_score = sum(avg_chamfer_score) / len(train_dataset)
        print("Finish epoch {}, loss {}, f score {}".format(epoch, tem, tem_f_score))

        val_loss, val_f_score, val_chamfer_score = val()

        # wandb.log(
        #     {
        #         "Train loss": tem,
        #         "Train F-score": tem_f_score,
        #         "Val loss": val_loss,
        #         "Val F-score": val_f_score,
        #         "Train chamfer matching score": tem_chamfer_score,
        #         "Val chamfer matching score": val_chamfer_score,
        #     }
        # )

        if pre_loss == None:
            torch.save(mymodel, "best_model_refine.pth")
            pre_loss = val_loss
        elif pre_loss > val_loss:
            torch.save(mymodel, "best_model_refine.pth")
            pre_loss = val_loss
        mymodel.train()
        if (epoch + 1) % 50 == 0:
            scheduler.step()


def val():
    mymodel.eval()
    avg_loss = []
    avg_f_score = []
    avg_chamfer_score = []
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
        ) = data

        predictMask = PSPmodel(input_img.cuda().float())

        predictMask = torch.argmax(predictMask, 1, keepdim=True)
        predictMask = (predictMask == 1).detach()

        inputData = torch.cat(
            (edge_img.cuda().float(), mask_img.cuda().float(), predictMask.float(), input_img.cuda().float()), 1
        )
        input = Variable(inputData)

        # load data to cuda
        init3dPt = init3dPt.cuda().float()
        target3dPt = target3dPt.cuda().float()
        initPose = initPose.cuda().float()
        targetPose = targetPose.cuda().float()
        rescaleValue = rescaleValue.cuda().float()

        with torch.no_grad():
            rot, trans, dist = mymodel(input)

        trans = trans.unsqueeze(1)
        dist = dist.unsqueeze(1)

        trans = (trans - 0.5) * input_img.shape[-1]

        # generate the rotation pose
        rot_pose = torch.bmm(initPose, tgm.angle_axis_to_rotation_matrix(rot))

        # update the camera matrix because the input image is resize
        camera_matrix_original = (
            torch.tensor(
                [
                    [
                        [CFG.CAMERA_MATRIX[0, 0], 0.0, float(input_img.shape[-2]) / 2,],
                        [0.0, CFG.CAMERA_MATRIX[1, 1], float(input_img.shape[-1]) / 2,],
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

        # get the target 2d points
        target3dPt = tgm.transform_points(targetPose, target3dPt)
        target_2d_pts = kornia.project_points(target3dPt, camera_matrix)

        # get the predict 2d points
        predict3dpt = tgm.transform_points(rot_pose, init3dPt)
        predict_2d_pts = kornia.project_points(predict3dpt, camera_matrix)

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
                trans[:, :, 0].view(trans.shape[0])
                * trans[:, :, 0].view(trans.shape[0])
                + torch.tensor(CFG.CAMERA_MATRIX[1, 1] * CFG.CAMERA_MATRIX[1, 1]).cuda()
                * rescaleValue
                * rescaleValue
            )
            * torch.tensor(CFG.CAMERA_MATRIX[1, 1] / CFG.CAMERA_MATRIX[0, 0]).cuda(),
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

        # predict 3d pts
        predict3dpts = tgm.transform_points(pred_pose, init3dPt)

        # apply the translation on 2d points
        center = torch.tensor(
            [[float(input_img.shape[-2]) / 2, float(input_img.shape[-1]) / 2]]
        ).cuda()

        # scale the 2d points with dist
        scaled_predict_2d_pts = torch.add(
            torch.add(predict_2d_pts, center * NEG_ONE_PAIR) * dist, center
        )

        # shift the 2d points with trans
        predict_2d_pts = torch.add(scaled_predict_2d_pts, trans)

        # calculate the chamfer distance loss
        dist1, dist2, idx1, idx2 = chamLoss(predict_2d_pts.float(), target_2d_pts.float())

        # get distance between each point pairs in 3d
        targetFromInit3dPt = tgm.transform_points(targetPose, init3dPt)
        distanceBetweenVec3d = predict3dpts - targetFromInit3dPt

        # get distance between each point pairs in 2d
        target_2d_pts_init = kornia.project_points(targetFromInit3dPt, camera_matrix)

        distanceBetweenVec = predict_2d_pts - target_2d_pts_init

        # get normal distance
        normdist2d = torch.norm(distanceBetweenVec, p=1, dim=2)
        normdist3d = torch.norm(distanceBetweenVec3d, p=2, dim=2)

        ch_loss = torch.mean(dist1, 1) + torch.mean(dist2, 1)
        loss = torch.mean(normdist3d, 1) * lambda3d + torch.mean(normdist2d, 1)

        avg_loss.append(loss.data.cpu().numpy().sum())
        f_score, precision, recall = FS.fscore(dist1, dist2)
        avg_f_score.append(f_score.data.cpu().numpy().sum())
        avg_chamfer_score.append(ch_loss.data.cpu().numpy().sum())

    tem = sum(avg_loss) / len(val_dataset)
    tem_f_score = sum(avg_f_score) / len(val_dataset)
    tem_chamfer_score = sum(avg_chamfer_score) / len(val_dataset)
    print("val loss {}".format(tem))
    return tem, tem_f_score, tem_chamfer_score


if __name__ == "__main__":
    train()
    print("done")
