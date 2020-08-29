# run testing on image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.models.train_refine_model import getPredictPose

from models.model import Refine_Net
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

    chamLoss2d = CHAMFER2D.chamfer_2DDist()
    chamLoss3d = CHAMFER3D.chamfer_3DDist()
    softmax = nn.Softmax2d()

    return mymodel, softmax, chamLoss2d, chamLoss3d


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

    flow_inputData = torch.cat((mask_img, edge_img, input_img), 1,)
    flow_input = Variable(flow_inputData)

    rot, trans, dist, opticalFlow, segmentMask = mymodel(flow_input)
    pred_pose = getPredictPose(
        initPose, rot, trans, dist, input_img.shape[-2], rescaleValue
    )

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

    # predict 3d pts
    predict3dpts = tgm.transform_points(pred_pose, init3dPt)
    predict_2d_pts = kornia.project_points(predict3dpts, camera_matrix)


if __name__ == "__main__":
    m, s, chamLoss2d, chamLoss3d = init()
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

