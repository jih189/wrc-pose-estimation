# run testing on image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from poseUtil import getPredictPose

from models.model import DeepIM
import torchgeometry as tgm
import kornia

import torch
import torch.nn as nn
from torchvision import utils

from poseUtil import getPredictPose, getRotationError, ADD_error, ADDS_error

from torch.autograd import Variable
from tqdm import tqdm
import src.configuration as CFG
import src.common.object_model as OM


def init():
    refine_model = DeepIM().cuda()

    refine_model.load_state_dict(torch.load(CFG.BEST_MODEL_ITERATIVE_REFINE))
    refine_model.eval()

    refine_model = nn.DataParallel(refine_model)
    return refine_model


def predict(mymodel, predict_index, view_image):

    OM.setup(CFG.CAMERA_W, CFG.CAMERA_H)
    OM.setProjectMatrixWithIntr(CFG.CAMERA_MATRIX, CFG.CAMERA_W, CFG.CAMERA_H)

    obj = OM.ObjectModel()
    obj.loadObjectCADModel(CFG.CAD_MODEL)
    obj.setIntrinsicMatrix(CFG.CAMERA_MATRIX)

    obj.determineSharpEdges(0.4)
    obj.generateSamplePoints(0.0003)

    numForTest = "{:06d}".format(predict_index)
    processed_data_dir = CFG.REFINE_ITERATIVE_DATA_PATH

    # load rgb image
    img_path = processed_data_dir + numForTest + "img.png"
    img = cv2.imread(img_path)

    # load init pose
    initPose_path = processed_data_dir + numForTest + "initPose.npy"
    initPose = np.load(initPose_path)

    for step in range(10):
        demo = img.copy()
        obj.setModelviewMatrix(initPose)
        obj.findVisibleSamplePoint()

        for p in obj.sharp_2d_pts:
            demo = cv2.circle(
                demo, (int(p[0]), int(p[1])), radius=1, color=(0, 0, 255), thickness=-1,
            )

        # get the horizontal rotation and vertical rotation
        horizontalR_ori, verticalR_ori = obj.getCenterAngle(initPose)

        # get init mask
        mask = obj.getVisibleArea()

        # get edge img
        edge = obj.getEdge(img.shape[0], img.shape[1])

        # find the crop size
        [x, y, w, h] = cv2.boundingRect(mask)

        boundingsize = max(w, h) * CFG.EXPAND_SIZE

        # get center point from pose
        centerPoint = obj.project3Dto2D((0, 0, 0), initPose)

        ex = int(centerPoint[0] - boundingsize / 2)
        ey = int(centerPoint[1] - boundingsize / 2)
        ew = int(boundingsize)
        eh = int(boundingsize)

        crop_img = np.zeros((eh, ew, 3), np.uint8,)
        crop_mask = np.zeros((eh, ew), np.uint8)
        crop_edge = np.zeros((eh, ew), np.uint8)

        upperleft_crop_inner = [max(0, ex), max(0, ey)]
        lowerright_crop_inner = [
            min(img.shape[1], ex + ew),
            min(img.shape[0], ey + eh),
        ]

        # cropped image with initial pose as center
        crop_img[
            upperleft_crop_inner[1] - ey : lowerright_crop_inner[1] - ey,
            upperleft_crop_inner[0] - ex : lowerright_crop_inner[0] - ex,
        ] = img[
            int(upperleft_crop_inner[1]) : int(lowerright_crop_inner[1]),
            int(upperleft_crop_inner[0]) : int(lowerright_crop_inner[0]),
        ].copy()

        crop_mask[
            upperleft_crop_inner[1] - ey : lowerright_crop_inner[1] - ey,
            upperleft_crop_inner[0] - ex : lowerright_crop_inner[0] - ex,
        ] = mask[
            int(upperleft_crop_inner[1]) : int(lowerright_crop_inner[1]),
            int(upperleft_crop_inner[0]) : int(lowerright_crop_inner[0]),
        ].copy()

        crop_edge[
            upperleft_crop_inner[1] - ey : lowerright_crop_inner[1] - ey,
            upperleft_crop_inner[0] - ex : lowerright_crop_inner[0] - ex,
        ] = edge[
            int(upperleft_crop_inner[1]) : int(lowerright_crop_inner[1]),
            int(upperleft_crop_inner[0]) : int(lowerright_crop_inner[0]),
        ].copy()

        # apply rotation on the initial pose to move to it to the center
        rough_pose_at_center = obj.rotatePoseWithAngle(
            initPose, horizontalR_ori, verticalR_ori
        )

        # calculate the resize scale
        rescaleValue = float(CFG.IMG_SIZE) / crop_img.shape[0]

        # resize rgb image
        crop_img = cv2.resize(
            crop_img, (CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=cv2.INTER_AREA
        )
        test_img = crop_img.copy()

        crop_img = crop_img[:, :, :3].transpose(2, 0, 1)

        crop_img = Variable(torch.from_numpy(crop_img)).float() / 255.0
        crop_img = crop_img.unsqueeze(0)

        # load edge image
        edge_img = cv2.resize(
            crop_edge, (CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=cv2.INTER_AREA
        )
        edge_img = edge_img[:, :, np.newaxis].transpose(2, 0, 1)

        edge_img = Variable(torch.from_numpy(edge_img)).float() / 255.0
        edge_img = edge_img.unsqueeze(0)

        # load the mask image
        mask_img = cv2.resize(
            crop_mask, (CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=cv2.INTER_AREA
        )
        mask_img = mask_img[:, :, np.newaxis].transpose(2, 0, 1)

        mask_img = Variable(torch.from_numpy(mask_img)).float() / 255.0
        mask_img = mask_img.unsqueeze(0)

        flow_inputData = torch.cat((mask_img, edge_img, crop_img), 1,)

        flow_input = Variable(flow_inputData).cuda()

        rot, trans, dist, opticalFlow, segmentMask = mymodel(flow_input)

        seg_pred = (
            torch.argmax(segmentMask, 1, keepdim=True)
            .float()
            .squeeze(1)
            .cpu()
            .detach()
            .numpy()
        )

        trans = trans.unsqueeze(1)
        dist = dist.unsqueeze(1)

        # update the camera matrix because the input image is resize
        camera_matrix_original = torch.tensor(
            [
                [
                    [CFG.CAMERA_MATRIX[0, 0], 0.0, float(crop_img.shape[-2]) / 2,],
                    [0.0, CFG.CAMERA_MATRIX[1, 1], float(crop_img.shape[-2]) / 2,],
                    [0.0, 0.0, 1.0],
                ]
            ]
        ).repeat(trans.shape[0], 1, 1)
        camera_matrix = camera_matrix_original.clone()
        camera_matrix[:, 0, 0] = camera_matrix_original[:, 0, 0] * rescaleValue
        camera_matrix[:, 1, 1] = camera_matrix_original[:, 1, 1] * rescaleValue
        camera_matrix.unsqueeze_(1)
        rough_pose_at_center = torch.from_numpy(rough_pose_at_center)
        rough_pose_at_center = rough_pose_at_center.unsqueeze(0)

        pred_pose = getPredictPose(
            rough_pose_at_center, rot, trans, dist, crop_img.shape[-2], rescaleValue,
        )

        pred_pose = obj.rotatePoseWithAngle(
            pred_pose[0].detach().cpu().numpy(), -horizontalR_ori, -verticalR_ori,
        )

        obj.setModelviewMatrix(pred_pose)
        obj.findVisibleSamplePoint()

        # test
        for p in obj.sharp_2d_pts:
            demo = cv2.circle(
                demo, (int(p[0]), int(p[1])), radius=1, color=(0, 255, 0), thickness=-1,
            )

        if view_image:
            cv2.imshow("test", demo)
            cv2.waitKey(0)

        initPose = pred_pose


if __name__ == "__main__":
    m = init()

    predict(m, 1400, True)
