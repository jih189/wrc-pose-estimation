import src.common.object_model as OM
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import src.configuration as CFG

import torchgeometry as tgm
import kornia

from models.models import Darknet  # set ONNX_EXPORT in models.py
from models.model import Magic_Net, FlowNet, DeepIM

from src.utils.utils import (
    load_classes,
    non_max_suppression,
    scale_coords,
    plot_one_box,
)
from poseUtil import getPredictPose, getConfid
import math

import cv2

import time

# ignore warming
np.seterr(divide="ignore", invalid="ignore")


def letterbox(
    img,
    new_shape=(416, 416),
    color=(128, 128, 128),
    auto=True,
    scaleFill=False,
    scaleup=True,
    interp=cv2.INTER_AREA,
):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = max(new_shape) / max(shape)
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(
            img, new_unpad, interpolation=interp
        )  # INTER_AREA is better, INTER_LINEAR is faster
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return img, ratio, (dw, dh)


if __name__ == "__main__":

    OM.setup(CFG.CAMERA_W, CFG.CAMERA_H)
    OM.setProjectMatrixWithIntr(CFG.CAMERA_MATRIX, CFG.CAMERA_W, CFG.CAMERA_H)

    obj = OM.ObjectModel()
    obj.loadObjectCADModel(CFG.CAD_MODEL)
    obj.setIntrinsicMatrix(CFG.CAMERA_MATRIX)

    obj.determineSharpEdges(0.8)
    obj.generateSamplePoints(0.0001)

    ###################### yolo ########################
    cfg = "cfg/yolov3-tiny3.cfg"
    weights = "weights/best_yolo_model_wrc.pt"
    conf_thres = 0.3
    iou_thres = 0.2
    device = "cuda"
    obj_names = "data/wrs-wrc.names"
    image_size = 416
    DIAG_PARAM = 1.0

    names = load_classes(obj_names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    yolo_model = Darknet(cfg, image_size)  # default image size is 416

    # Load weights
    yolo_model.load_state_dict(torch.load(weights)["model"])
    # Eval mode
    yolo_model.to(device).eval()

    ################### magic net ########################
    viewpt_class = CFG.VIEWPOINT_NUM
    rot_class = 60

    rot_model = Magic_Net(viewpt_class=viewpt_class, rot_class=rot_class).cuda()
    rot_model = torch.load(CFG.BEST_MODEL_ROT)
    # torch.save(rot_model.module.state_dict(), "best_model_rot_pulley-test.pth")
    rot_model.eval()

    ################# refine net ###########################
    refine_model = DeepIM().cuda()
    # refine_model.load_state_dict(
    #     torch.load("best_model_iterative_refine_pulley-test.pth")
    # )
    refine_model = torch.load(CFG.BEST_MODEL_ITERATIVE_REFINE)
    # torch.save(
    #     refine_model.module.state_dict(), "best_model_iterative_refine_pulley-test.pth"
    # )
    refine_model.eval()

    # read image
    frame = cv2.imread("input-sbar3.png")
    demo = frame.copy()
    rot_frame = frame.copy()
    refine_frame = frame.copy()

    # resize image
    frame = letterbox(frame, new_shape=416)[0]
    # frame = cv2.resize(frame, (int(320), int(416)), interpolation=cv2.INTER_AREA)
    frame = frame[:, :, :3]
    frame = frame[:, :, ::-1].transpose(2, 0, 1)
    frame = np.ascontiguousarray(frame)
    # load image to the device
    frame = torch.from_numpy(frame).to(device)

    # convert image to be used
    frame = frame.float()  # uint8 to fp16/32
    frame /= 255.0  # 0 - 255 to 0.0 - 1.0
    if frame.ndimension() == 3:
        frame = frame.unsqueeze(0)

    # Inference
    pred = yolo_model(frame)[0].float()

    # Apply NMS
    pred = non_max_suppression(
        pred, conf_thres, iou_thres, classes=None, agnostic=False
    )

    pred = pred[0]
    croptopleft, croplowright = None, None

    foundObject = False

    # Process detections
    if pred is not None and len(pred):
        # Rescale boxes from img_size to demo size
        pred[:, :4] = scale_coords(frame.shape[2:], pred[:, :4], demo.shape).round()

        for *xyxy, conf, cls in pred:
            label = "%s %.2f" % (names[int(cls)], conf)
            # plot_one_box(xyxy, demo, label=label, color=colors[int(cls)])
            if names[int(cls)] == "Pulley":
                foundObject = True
                croptopleft = [
                    int(xyxy[0].cpu().detach().numpy()),
                    int(xyxy[1].cpu().detach().numpy()),
                ]
                croplowright = [
                    int(xyxy[2].cpu().detach().numpy()),
                    int(xyxy[3].cpu().detach().numpy()),
                ]

    if foundObject:
        # rough_pose_estimation_start = time.time()
        # find the diagnal of the bounding box
        objectDiag = math.sqrt(
            (croplowright[0] - croptopleft[0]) ** 2
            + (croplowright[1] - croptopleft[1]) ** 2
        )

        # rot classifier
        upperleft, lowerright = OM.get_centered_crop(croptopleft, croplowright)
        crop_width = int(lowerright[0]) - int(upperleft[0])

        img_crop = np.zeros(
            (lowerright[1] - upperleft[1], lowerright[0] - upperleft[0], 3), np.uint8,
        )
        upperleft_crop_inner = [
            max(0, upperleft[0]),
            max(0, upperleft[1]),
        ]
        lowerright_crop_inner = [
            min(rot_frame.shape[1], lowerright[0]),
            min(rot_frame.shape[0], lowerright[1]),
        ]
        img_crop[
            upperleft_crop_inner[1]
            - upperleft[1] : lowerright_crop_inner[1]
            - upperleft[1],
            upperleft_crop_inner[0]
            - upperleft[0] : lowerright_crop_inner[0]
            - upperleft[0],
        ] = rot_frame[
            int(upperleft_crop_inner[1]) : int(lowerright_crop_inner[1]),
            int(upperleft_crop_inner[0]) : int(lowerright_crop_inner[0]),
        ]

        img_crop = cv2.resize(img_crop, (CFG.IMG_SIZE, CFG.IMG_SIZE))
        img_crop = img_crop[:, :, :3].transpose(2, 0, 1)
        img_crop = img_crop[np.newaxis, ...]

        input = Variable(torch.from_numpy(img_crop).cuda()).float()
        output = rot_model(input)

        rot_pred = output[:, :viewpt_class].data.cpu().numpy()
        rot_pred = np.argmax(rot_pred, axis=1)
        viewpt = np.array(OM.idx2vp(rot_pred[0]))

        rot = output[:, viewpt_class : viewpt_class + rot_class].data.cpu().numpy()
        rot = np.argmax(rot, axis=1)
        rot = rot[0] * np.pi / 30

        principle_pt = np.array([CFG.CAMERA_MATRIX[0, 2], CFG.CAMERA_MATRIX[1, 2]])

        position = (
            torch.sigmoid(output[:, viewpt_class + rot_class :]).data.cpu().numpy()
        )
        position *= crop_width
        offset = position[:, :2]
        offset = np.array(upperleft) + offset.reshape(2) - principle_pt

        # get the rough pose from view point, rotation, offset, and depth
        rough_pred_pose = obj.label2pose(viewpt, rot, offset, 0.5)

        temp_demo = demo.copy()

        obj.setModelviewMatrix(rough_pred_pose)
        obj.renderVisibleFaces()
        bx, by, w_temp, h_temp = cv2.boundingRect(obj.getVisibleArea())

        currentDiag = math.sqrt(w_temp ** 2 + h_temp ** 2)

        rough_pred_pose = obj.label2pose(
            viewpt, rot, offset, 0.5 * currentDiag / objectDiag * DIAG_PARAM
        )

        obj.setModelviewMatrix(rough_pred_pose)
        obj.findVisibleSamplePoint()

        # test
        for p in obj.sharp_2d_pts:
            temp_demo = cv2.circle(
                temp_demo,
                (int(p[0]), int(p[1])),
                radius=1,
                color=(255, 0, 0),
                thickness=-1,
            )
        cv2.imshow("test", temp_demo)
        cv2.waitKey(0)

        # rough_pose_estimation_end = time.time()
        # print("time for rought pose estimation: ")
        # print(rough_pose_estimation_end - rough_pose_estimation_start)

        numOfRefine = 5

        # pose refinement
        for t in range(numOfRefine):
            pose_refinement_start = time.time()
            obj.setModelviewMatrix(rough_pred_pose)
            # find_visible_sample_points_start = time.time()
            obj.findVisibleSamplePoint()
            # find_visible_sample_points_end = time.time()
            # print("time for visible points:")
            # print(find_visible_sample_points_end - find_visible_sample_points_start)

            horizontalR_ori, verticalR_ori = obj.getCenterAngle(rough_pred_pose)

            # get init mask
            mask = obj.getVisibleArea()

            # get edge img
            edge = obj.getEdge(refine_frame.shape[0], refine_frame.shape[1])

            # find the crop size
            [x, y, w, h] = cv2.boundingRect(mask)

            boundingsize = max(w, h) * CFG.EXPAND_SIZE

            # get center point from pose
            centerPoint = obj.project3Dto2D((0, 0, 0), rough_pred_pose)

            ex = int(centerPoint[0] - boundingsize / 2)
            ey = int(centerPoint[1] - boundingsize / 2)
            ew = int(boundingsize)
            eh = int(boundingsize)

            crop_img = np.zeros((eh, ew, 3), np.uint8,)
            crop_mask = np.zeros((eh, ew), np.uint8)
            crop_edge = np.zeros((eh, ew), np.uint8)

            upperleft_crop_inner = [max(0, ex), max(0, ey)]
            lowerright_crop_inner = [
                min(refine_frame.shape[1], ex + ew),
                min(refine_frame.shape[0], ey + eh),
            ]

            # cropped image with initial pose as center
            crop_img[
                upperleft_crop_inner[1] - ey : lowerright_crop_inner[1] - ey,
                upperleft_crop_inner[0] - ex : lowerright_crop_inner[0] - ex,
            ] = refine_frame[
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
                crop_mask, (CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=cv2.INTER_AREA
            )
            mask_img = mask_img[:, :, np.newaxis].transpose(2, 0, 1)

            mask_img = Variable(torch.from_numpy(mask_img)).float() / 255.0
            mask_img = mask_img.unsqueeze(0)

            flow_inputData = torch.cat((mask_img, edge_img, crop_img), 1,)

            flow_input = Variable(flow_inputData).cuda()

            rot, trans, dist, opticalFlow, segmentMask = refine_model(flow_input)

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

            if t == numOfRefine - 1:
                # get the confidence
                # confidence_estimation_time_start = time.time()
                confidence = getConfid(segmentMask, opticalFlow, mask_img,)
                # confidence_estimation_time_end = time.time()
                # print("confidence estimation time:")
                # print(confidence_estimation_time_end - confidence_estimation_time_start)
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

            rough_pred_pose = pred_pose
            pose_refinement_end = time.time()
            print("time for pose refinement: ")
            print(pose_refinement_end - pose_refinement_start)
            rough_pred_pose = OM.symmetricRemove(rough_pred_pose)

            temp_demo = demo.copy()
            obj.setModelviewMatrix(rough_pred_pose)
            obj.findVisibleSamplePoint()

            # test
            for p in obj.sharp_2d_pts:
                temp_demo = cv2.circle(
                    temp_demo,
                    (int(p[0]), int(p[1])),
                    radius=1,
                    color=(0, 255, 0),
                    thickness=-1,
                )
            cv2.imshow("test", temp_demo)
            cv2.waitKey(1)

    else:
        print("can't find object!!")
    cv2.destroyAllWindows()
