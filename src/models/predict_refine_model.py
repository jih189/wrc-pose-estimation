# run testing on image
import numpy as np
import cv2
import matplotlib.pyplot as plt

from models.model import Refine_Net, PSPNet
import torchgeometry as tgm
import kornia

import torch
import torch.nn as nn
from torchvision import utils
import chamfer2D.dist_chamfer_2D as CHAMFER2D
import chamfer3D.dist_chamfer_3D as CHAMFER3D

from torch.autograd import Variable
from tqdm import tqdm
import src.configuration as CFG

IMG_SIZE = 240


def init():

    mymodel = Refine_Net().cuda()

    mymodel = nn.DataParallel(mymodel)
    mymodel = torch.load("best_model_refine_housing.pth")
    mymodel.eval()

    PSPmodel = PSPNet().cuda()
    PSPmodel = nn.DataParallel(PSPmodel)
    PSPmodel = torch.load("best_model_psp_housing.pth")
    PSPmodel.eval()

    chamLoss2d = CHAMFER2D.chamfer_2DDist()
    chamLoss3d = CHAMFER3D.chamfer_3DDist()
    softmax = nn.Softmax2d()

    return mymodel, PSPmodel, softmax, chamLoss2d, chamLoss3d


def predict(
    mymodel, pspmodel, softmax, predict_index, chamLoss2d, chamLoss3d, view_image
):
    numForTest = "{:06d}".format(predict_index)
    processed_data_dir = CFG.REFINE_SATA_PATH

    # load rgb image
    img_path = processed_data_dir + numForTest + "img.png"
    img = cv2.imread(img_path)

    # calculate the resize scale
    rescaleValue = float(IMG_SIZE) / img.shape[0]

    # resize rgb image
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    testimg = img.copy()
    preimg = img.copy()
    img = img[:, :, :3].transpose(2, 0, 1)

    img = Variable(torch.from_numpy(img).cuda()).float()
    img = img.unsqueeze(0)

    # load edge image
    edge_path = processed_data_dir + numForTest + "edge.png"
    edge_img = cv2.imread(edge_path)
    edge_img = cv2.resize(edge_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    edge_img = edge_img[:, :, :1].transpose(2, 0, 1)

    edge_img = Variable(torch.from_numpy(edge_img).cuda()).float()
    edge_img = edge_img.unsqueeze(0)

    predictMask = pspmodel(img)
    predictMask = torch.argmax(predictMask, 1, keepdim=True).float()
    predictMask = (predictMask == 1).float().detach()

    # load bounding image
    # bounding_path = processed_data_dir + numForTest + "bounding.png"
    # bounding_img = cv2.imread(bounding_path)
    # bounding_img = cv2.resize(
    #     bounding_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
    # )
    # bounding_img = bounding_img[:, :, :1].transpose(2, 0, 1)

    # bounding_img = Variable(torch.from_numpy(bounding_img).cuda()).float()
    # bounding_img = bounding_img.unsqueeze(0)

    # load the mask image
    mask_path = processed_data_dir + numForTest + "mask.png"
    mask_img = cv2.imread(mask_path)
    mask_img = cv2.resize(mask_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    mask_img = mask_img[:, :, :1].transpose(2, 0, 1)

    mask_img = Variable(torch.from_numpy(mask_img).cuda()).float()
    mask_img = mask_img.unsqueeze(0)

    # load the init 3d points
    init3dPt_path = processed_data_dir + numForTest + "init3dPt.npy"
    init3dPt = np.load(init3dPt_path)

    init3dPt = Variable(torch.from_numpy(init3dPt).cuda()).float()
    init3dPt = init3dPt.unsqueeze(0)

    # load init pose
    initPose_path = processed_data_dir + numForTest + "initPose.npy"
    initPose = np.load(initPose_path)
    initPose = Variable(torch.from_numpy(initPose).cuda()).float()
    initPose = initPose.unsqueeze(0)

    # load the target 3d points
    target3dPt_path = processed_data_dir + numForTest + "target3dPt.npy"
    target3dPt = np.load(target3dPt_path)

    target3dPt = Variable(torch.from_numpy(target3dPt).cuda()).float()
    target3dPt = target3dPt.unsqueeze(0)

    # load the target pose
    targetPose_path = processed_data_dir + numForTest + "targetPose.npy"
    targetPose = np.load(targetPose_path)
    targetPose = Variable(torch.from_numpy(targetPose).cuda()).float()
    targetPose = targetPose.unsqueeze(0)

    # running model
    inputData = torch.cat((edge_img, mask_img, predictMask, img), 1)
    rot, trans, dist = mymodel(inputData)

    trans = trans.unsqueeze(1)
    trans = (trans - 0.5) * 240

    dist = dist.unsqueeze(1)

    # generate the rotation pose
    # pose = torch.bmm(initPose, tgm.angle_axis_to_rotation_matrix(rot))

    initPoseRot = torch.eye(4).repeat(trans.shape[0], 1, 1).cuda().float()
    initPoseRot[:, :3, :3] = initPose[:, :3, :3]

    # generate the rotation pose
    pose = torch.bmm(tgm.angle_axis_to_rotation_matrix(rot), initPoseRot)
    pose[:, :3, 3] = initPose[:, :3, 3]

    # update the camera matrix because the input image is resize
    camera_matrix = torch.tensor(
        [
            [
                [CFG.CAMERA_MATRIX[0, 0] * rescaleValue, 0.0, float(img.shape[-2]) / 2],
                [0.0, CFG.CAMERA_MATRIX[1, 1] * rescaleValue, float(img.shape[-1]) / 2],
                [0.0, 0.0, 1.0],
            ]
        ]
    ).cuda()

    # get the predict 2d points
    predict3dpt = tgm.transform_points(pose, init3dPt)

    #############################################################
    dist_pose = pose.clone()
    dist_pose[:, 2, 3] = pose[:, 2, 3] / dist[:, 0, 0]

    horizontalR = torch.atan2(
        trans[:, :, 0], torch.tensor(654.968116289191 * rescaleValue).cuda()
    )
    verticalR = -torch.atan2(
        trans[:, :, 1],
        torch.sqrt(
            trans[:, :, 0] * trans[:, :, 0]
            + torch.tensor(
                657.1436336052552 * 657.1436336052552 * rescaleValue * rescaleValue
            ).cuda()
        )
        * torch.tensor(657.1436336052552 / 654.968116289191).cuda(),
    )

    ch = torch.cos(horizontalR)
    sh = torch.sin(horizontalR)
    ca = torch.cos(torch.tensor([[0.0]]).cuda())  # not this
    sa = torch.sin(torch.tensor([[0.0]]).cuda())  # not this
    cb = torch.cos(verticalR)
    sb = torch.sin(verticalR)

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
    rotation_matrix[..., :3, :3] = torch.cat(
        [m00, m01, m02, m10, m11, m12, m20, m21, m22], dim=1
    ).view(-1, 3, 3)

    rot_pose = torch.bmm(rotation_matrix, dist_pose)

    predict3dpt_real_rot = tgm.transform_points(rot_pose, init3dPt)

    test_2d_pts = kornia.project_points(predict3dpt_real_rot, camera_matrix)

    #############################################

    predict_2d_pts = kornia.project_points(predict3dpt, camera_matrix)

    center = torch.tensor([[float(img.shape[-2]) / 2, float(img.shape[-1]) / 2]]).cuda()

    neg_one_pair = torch.tensor([[-1.0, -1.0]]).cuda()
    scaled_predict_2d_pts = torch.add(
        torch.add(predict_2d_pts, center * neg_one_pair) * dist, center
    )
    predict_2d_pts = torch.add(scaled_predict_2d_pts, trans)

    origin3dpt = tgm.transform_points(initPose, init3dPt)
    origin_2d_pts = kornia.project_points(origin3dpt, camera_matrix)

    label3dpt = tgm.transform_points(targetPose, target3dPt)
    label_2d_pts = kornia.project_points(label3dpt, camera_matrix)

    # edge of init pose
    for p in origin_2d_pts.cpu().detach().numpy()[0]:
        preimg = cv2.circle(
            preimg, (int(p[0]), int(p[1])), radius=0, color=(255, 0, 0), thickness=-1
        )

    # edge of predicted pose
    for p in predict_2d_pts.cpu().detach().numpy()[0]:
        preimg = cv2.circle(
            preimg, (int(p[0]), int(p[1])), radius=0, color=(0, 255, 0), thickness=-1
        )

    # # edge of label pose
    # for p in label_2d_pts.cpu().detach().numpy()[0]:
    #     preimg = cv2.circle(
    #         preimg, (int(p[0]), int(p[1])), radius=0, color=(0, 0, 255), thickness=-1
    #     )

    if view_image:
        preimg = cv2.resize(preimg, (0, 0), fx=5, fy=5)
        cv2.imshow("test", preimg)
        cv2.waitKey(0)
        return None
    else:
        dist1, dist2, idx1, idx2 = chamLoss2d(
            predict_2d_pts.float(), label_2d_pts.float()
        )
        ch_loss2d = torch.mean(dist1, 1) + torch.mean(dist2, 1)
        dist1, dist2, idx1, idx2 = chamLoss2d(predict3dpt, label3dpt)
        ch_loss3d = torch.mean(dist1, 1)  # + torch.mean(dist2, 1)

        return ch_loss2d.cpu().detach().numpy()[0], ch_loss3d.cpu().detach().numpy()[0]


if __name__ == "__main__":
    m, p, s, chamLoss2d, chamLoss3d = init()
    highestLoss = 0.0
    highestIndex = None
    diagonalDist = 0.0335 * 1 * 0.1
    correct = 0

    predict(m, p, s, 375, chamLoss2d, chamLoss3d, True)
    # for i in tqdm(range(2400)):
    #     ch_loss2d, ch_loss3d = predict(m, p, s, i, chamLoss2d, chamLoss3d, False)
    #     if ch_loss3d < diagonalDist:
    #         correct += 1
    #     if ch_loss2d > highestLoss:
    #         highestIndex = i
    #         highestLoss = ch_loss2d
    # print("add-s: ", correct / 9000)
    # print("worst case: ", highestIndex)

