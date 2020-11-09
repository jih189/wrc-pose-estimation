import src.common.object_model as OM
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import src.configuration as CFG
from pathlib import Path

import torchgeometry as tgm
import kornia

from models.models import Darknet  # set ONNX_EXPORT in models.py
from models.model import Magic_Net, DeepIM, FlowNet
from torch.multiprocessing import Pool, Value, cpu_count, Array

torch.multiprocessing.set_sharing_strategy("file_system")
import math

from src.utils.utils import (
    load_classes,
    non_max_suppression,
    scale_coords,
    plot_one_box,
)
import tqdm
import cv2
from poseUtil import getPredictPose, ADD_error, getConfid

counter = Value("i", 0)

viewpt_class = CFG.VIEWPOINT_NUM
rot_class = 60
DIAG_PARAM = 0.5

# ignore warming
np.seterr(divide="ignore", invalid="ignore")

totalADD = []
totalADDS = []


def init():

    ################## rot model ################################

    rot_model = Magic_Net(viewpt_class=viewpt_class, rot_class=rot_class).cpu()
    rot_model = torch.load(CFG.BEST_MODEL_ROT)
    rot_model.eval()
    rot_model.share_memory()

    ############### refine model ##########################
    refine_model = DeepIM()
    refine_model = torch.load(CFG.BEST_MODEL_ITERATIVE_REFINE)
    refine_model.eval()
    refine_model.share_memory()

    return None, rot_model, refine_model


def obj_init():
    OM.setup(CFG.CAMERA_W, CFG.CAMERA_H)
    OM.setProjectMatrixWithIntr(CFG.CAMERA_MATRIX, CFG.CAMERA_W, CFG.CAMERA_H)

    obj = OM.ObjectModel()
    obj.loadObjectCADModel(CFG.CAD_MODEL)
    obj.setIntrinsicMatrix(CFG.CAMERA_MATRIX)

    obj.determineSharpEdges(0.8)
    obj.generateSamplePoints(0.00001, 0.00001)
    return obj


# generate a bounding box according to the object and its pose with noise
def generateBoundingbox(obj, pose):
    depth = pose[2, 3]
    obj.setModelviewMatrix(pose)
    obj.findVisibleSamplePoint()

    # extract the bounding box
    bx, by, bw, bh = cv2.boundingRect(obj.getVisibleArea())
    upperleft = np.array([bx, by])
    lowerright = np.array([bx + bw, by + bh])

    high = np.clip(10 / depth, 0, 100)
    upperleft = (upperleft - np.random.uniform(0, high, upperleft.shape)).astype(np.int)
    lowerright = (lowerright + np.random.uniform(0, high, lowerright.shape)).astype(
        np.int
    )

    crop_upperleft, crop_lowerright = OM.get_centered_crop(upperleft, lowerright)

    predictDiag = math.sqrt(bw ** 2 + bh ** 2)
    return True, predictDiag, crop_upperleft, crop_lowerright


def poseEstimationWithFlow(
    obj,
    rough_pred_pose,
    segmentMask,
    opticalFlow,
    mask_img,
    test_img,
    ex,
    ey,
    rescaleValue,
):
    seg_pred = (
        torch.argmax(segmentMask, 1, keepdim=True)
        .float()
        .squeeze(1)
        .cpu()
        .detach()
        .numpy()
    )
    cv2.imshow("mask", seg_pred[0])
    obj.setModelviewMatrix(rough_pred_pose)
    obj.findVisibleSamplePoint()
    obj.getVisiblePointCloud()

    opticalFlow = torch.sigmoid(opticalFlow)

    # padding = Variable(
    #     torch.zeros(opticalFlow.shape[0], 1, opticalFlow.shape[2], opticalFlow.shape[3])
    # ).cuda()

    # opticalFlow = torch.cat((opticalFlow, padding), 1)

    opticalFlow = opticalFlow * (mask_img.cuda() == 1.0)

    objectPoints = []
    imagePoints = []

    for i in range(len(obj.pointcloud)):
        y2d, x2d, x3d, y3d, z3d = obj.pointcloud[i]
        y2d = (y2d - ey) * rescaleValue
        x2d = (x2d - ex) * rescaleValue
        [mx, my] = opticalFlow[0, :2, int(y2d), int(x2d)].cpu().detach().numpy()
        if mx != 0.0 or my != 0.0:
            mx = (mx - 0.5) * CFG.IMG_SIZE
            my = (my - 0.5) * CFG.IMG_SIZE
            if (
                x2d + mx >= 0
                and x2d + mx < CFG.IMG_SIZE
                and y2d + my >= 0
                and y2d + my < CFG.IMG_SIZE
                and seg_pred[0, int(y2d + my), int(x2d + mx)] == 1.0
                and int(x2d) % 10 == 0
                and int(y2d) % 10 == 0
            ):
                objectPoints.append([x3d, y3d, z3d])
                imagePoints.append(
                    [(x2d + mx) / rescaleValue + ex, (y2d + my) / rescaleValue + ey,]
                )
                test_img = cv2.line(
                    test_img,
                    (int(x2d + mx), int(y2d + my)),
                    (int(x2d), int(y2d)),
                    (0, 255, 255),
                    1,
                )
                test_img = cv2.circle(
                    test_img,
                    (int(x2d + mx), int(y2d + my)),
                    radius=1,
                    color=(255, 0, 0),
                    thickness=-1,
                )
    cv2.imshow("test", test_img)

    # objectPoints = np.array(objectPoints)
    # imagePoints = np.array(imagePoints)

    # _, rvec, tvec = cv2.solvePnP(
    #     objectPoints,
    #     imagePoints,
    #     CFG.CAMERA_MATRIX,
    #     np.zeros((4, 1)),
    #     flags=cv2.SOLVEPNP_EPNP,
    # )

    # rotMat, _ = cv2.Rodrigues(rvec)
    # pnppose = np.identity(4)
    # pnppose[:3, :3] = rotMat
    # pnppose[0, 3] = tvec[0][0]
    # pnppose[1, 3] = tvec[1][0]
    # pnppose[2, 3] = tvec[2][0]

    # return pnppose


def testData(obj, yolo_model, rot_model, refine_model, d):

    image_name, pose_name = d
    # read image
    img = cv2.imread(image_name)
    rot_frame = img.copy()
    refine_frame = img.copy()
    demo = img.copy()

    # read the pose
    targetPose = np.load(pose_name)
    targetPose_temp = targetPose.copy()

    # use obj model to generate the bounding box
    foundObject, predictDiag, croptopleft, croplowright = generateBoundingbox(
        obj, targetPose
    )

    targetPose = torch.from_numpy(targetPose)

    if foundObject:

        img_crop = np.zeros(
            (croplowright[1] - croptopleft[1], croplowright[0] - croptopleft[0], 3,),
            np.uint8,
        )
        upperleft_crop_inner = [
            max(0, croptopleft[0]),
            max(0, croptopleft[1]),
        ]
        lowerright_crop_inner = [
            min(img.shape[1], croplowright[0]),
            min(img.shape[0], croplowright[1]),
        ]
        img_crop[
            upperleft_crop_inner[1]
            - croptopleft[1] : lowerright_crop_inner[1]
            - croptopleft[1],
            upperleft_crop_inner[0]
            - croptopleft[0] : lowerright_crop_inner[0]
            - croptopleft[0],
        ] = img[
            int(upperleft_crop_inner[1]) : int(lowerright_crop_inner[1]),
            int(upperleft_crop_inner[0]) : int(lowerright_crop_inner[0]),
        ]

        crop_width = int(croplowright[0]) - int(croptopleft[0])

        # resize the image so rot classifier can process
        img_crop = cv2.resize(img_crop, (CFG.IMG_SIZE, CFG.IMG_SIZE))

        img_crop = img_crop[:, :, :3].transpose(2, 0, 1)
        img_crop = img_crop[np.newaxis, ...]

        input = Variable(torch.from_numpy(img_crop)).float()
        output = rot_model(input)

        rot_pred = output[:, :viewpt_class].cpu().data.numpy()
        rot_pred = np.argmax(rot_pred, axis=1)
        viewpt = np.array(OM.idx2vp(rot_pred[0]))

        rot = output[:, viewpt_class : viewpt_class + rot_class].cpu().data.numpy()
        rot = np.argmax(rot, axis=1)
        rot = rot[0] * np.pi / 30

        principle_pt = np.array([CFG.CAMERA_MATRIX[0, 2], CFG.CAMERA_MATRIX[1, 2]])

        position = (
            torch.sigmoid(output[:, viewpt_class + rot_class :]).cpu().data.numpy()
        )
        position *= crop_width
        offset = position[:, :2]
        offset = np.array(croptopleft) + offset.reshape(2) - principle_pt

        # get the rough pose from view point, rotation, offset, and depth
        rough_pred_pose = obj.label2pose(viewpt, rot, offset, 0.5)

        obj.setModelviewMatrix(rough_pred_pose)
        obj.renderVisibleFaces()
        _, _, w_temp, h_temp = cv2.boundingRect(obj.getVisibleArea())
        currentDiag = math.sqrt(w_temp ** 2 + h_temp ** 2)

        rough_pred_pose = obj.label2pose(
            viewpt, rot, offset, 0.5 * predictDiag / currentDiag * DIAG_PARAM
        )

        numOfRefine = 10

        # # pose refinement
        for t in range(numOfRefine):
            obj.setModelviewMatrix(rough_pred_pose)
            obj.findVisibleSamplePoint()

            refine_demo = demo.copy()

            # # draw image
            for p in obj.sharp_2d_pts:
                p = (int(p[0]), int(p[1]))
                refine_demo = cv2.circle(
                    refine_demo, p, radius=0, color=(0, 0, 255), thickness=-1
                )

            horizontalR_ori, verticalR_ori = obj.getCenterAngle(rough_pred_pose)

            # get init mask
            init_mask = obj.getVisibleArea()

            # get edge img
            edge = obj.getEdge(refine_frame.shape[0], refine_frame.shape[1])

            # find the crop size
            [x, y, w, h] = cv2.boundingRect(init_mask)

            boundingsize = max(w, h) * CFG.EXPAND_SIZE

            # get center point from pose
            centerPoint = obj.project3Dto2D((0, 0, 0), rough_pred_pose)

            ex = int(centerPoint[0] - boundingsize / 2)
            ey = int(centerPoint[1] - boundingsize / 2)
            ew = int(boundingsize)
            eh = int(boundingsize)

            crop_img = np.zeros((eh, ew, 3), np.uint8,)
            crop_init_mask = np.zeros((eh, ew), np.uint8)
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
            ]

            crop_init_mask[
                upperleft_crop_inner[1] - ey : lowerright_crop_inner[1] - ey,
                upperleft_crop_inner[0] - ex : lowerright_crop_inner[0] - ex,
            ] = init_mask[
                int(upperleft_crop_inner[1]) : int(lowerright_crop_inner[1]),
                int(upperleft_crop_inner[0]) : int(lowerright_crop_inner[0]),
            ]

            crop_edge[
                upperleft_crop_inner[1] - ey : lowerright_crop_inner[1] - ey,
                upperleft_crop_inner[0] - ex : lowerright_crop_inner[0] - ex,
            ] = edge[
                int(upperleft_crop_inner[1]) : int(lowerright_crop_inner[1]),
                int(upperleft_crop_inner[0]) : int(lowerright_crop_inner[0]),
            ]

            # apply rotation on the initial pose to move to it to the center
            rough_pose_at_center = obj.rotatePoseWithAngle(
                rough_pred_pose, horizontalR_ori, verticalR_ori
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
                crop_init_mask,
                (CFG.IMG_SIZE, CFG.IMG_SIZE),
                interpolation=cv2.INTER_AREA,
            )
            cv2.imshow("mask img", mask_img)
            mask_img = mask_img[:, :, np.newaxis].transpose(2, 0, 1)

            mask_img = Variable(torch.from_numpy(mask_img)).float() / 255.0
            mask_img = mask_img.unsqueeze(0)

            flow_inputData = torch.cat((mask_img, edge_img, crop_img), 1,)

            flow_input = Variable(flow_inputData)

            rot, trans, dist, opticalFlow, segmentMask = refine_model(flow_input)

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

            # poseEstimationWithFlow(
            #     obj,
            #     rough_pred_pose,
            #     segmentMask,
            #     opticalFlow,
            #     mask_img,
            #     test_img,
            #     ex,
            #     ey,
            #     rescaleValue,
            # )

            # if t == numOfRefine - 1:
            # get the confidence
            confidence = getConfid(segmentMask, opticalFlow, mask_img,)
            print("confidence ", confidence)

            pred_pose = getPredictPose(
                rough_pose_at_center,
                rot,
                trans,
                dist,
                crop_img.shape[-2],
                rescaleValue,
            )
            pred_pose = obj.rotatePoseWithAngle(
                pred_pose[0].detach().cpu().numpy(), -horizontalR_ori, -verticalR_ori,
            )

            # refine_demo = demo.copy()
            # obj.setModelviewMatrix(pred_pose)
            # obj.findVisibleSamplePoint()

            # # draw image
            # for p in obj.sharp_2d_pts:
            #     p = (int(p[0]), int(p[1]))
            #     refine_demo = cv2.circle(
            #         refine_demo, p, radius=0, color=(0, 0, 255), thickness=-1
            #     )

            cv2.imshow("demo", refine_demo)
            cv2.waitKey(0)

            rough_pred_pose = pred_pose


if __name__ == "__main__":
    yolo_model, rot_model, refine_model = init()

    input_filepath = CFG.VERIFY_IMAGE_PATH

    # read the test file and pose
    input_path = Path(input_filepath)
    image_names, pose_names = [], []
    for f in input_path.iterdir():
        if f.match("*.png"):
            image_names.append(str(f))
        if f.match("*.npy"):
            pose_names.append(str(f))
    image_names.sort()
    pose_names.sort()

    indexnum = 60
    image_names = image_names[indexnum : indexnum + 1]
    pose_names = pose_names[indexnum : indexnum + 1]

    datalist = list(zip(image_names, pose_names))

    obj = obj_init()
    for d in tqdm.tqdm(datalist):
        testData(obj, yolo_model, rot_model, refine_model, d)

    # print("average add is ", sum(totalADD) / len(totalADD))
    # print("average adds is ", sum(totalADDS) / len(totalADDS))

