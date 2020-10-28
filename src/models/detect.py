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
from src.models.poseUtil import getPredictPose

import cv2

# ignore warming
np.seterr(divide="ignore", invalid="ignore")

OM.setup(CFG.CAMERA_W, CFG.CAMERA_H)
OM.setProjectMatrixWithIntr(CFG.CAMERA_MATRIX, CFG.CAMERA_W, CFG.CAMERA_H)

obj = OM.ObjectModel()
obj.loadObjectCADModel(CFG.CAD_MODEL)
obj.setIntrinsicMatrix(CFG.CAMERA_MATRIX)

obj.determineSharpEdges(0.5)
obj.generateSamplePoints(0.0001, 0.000001)

###################### yolo ########################
webcam = "4"
cfg = "/home/cogrob-wrc/wrc-pose-estimation/cfg/yolov3-tiny3.cfg"
weights = "/home/cogrob-wrc/wrc-pose-estimation/weights/best_yolo_model_wrc.pt"
conf_thres = 0.7
iou_thres = 0.4
device = "cuda"
obj_names = "/home/cogrob-wrc/wrc-pose-estimation/data/wrs-wrc.names"
image_size = 416

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
rot_model.eval()

################# refine net ###########################
refine_model = DeepIM()
refine_model = torch.load(CFG.BEST_MODEL_ITERATIVE_REFINE)
refine_model.eval()


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


def detect(object_id, img, estimated_depth):

    frame = img
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
            plot_one_box(xyxy, demo, label=label, color=colors[int(cls)])
            if names[int(cls)] == object_id:
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
        # rot classifier
        upperleft, lowerright = OM.get_centered_crop(croptopleft, croplowright)
        l = int(lowerright[0]) - int(upperleft[0])

        # crop the image for rot classifier
        img_crop = rot_frame[
            int(upperleft[1]) : int(lowerright[1]),
            int(upperleft[0]) : int(lowerright[0]),
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
        position *= l
        offset = position[:, :2]
        offset = np.array(upperleft) + offset.reshape(2) - principle_pt

        depth = estimated_depth
        rough_pred_pose = obj.label2pose(viewpt, rot, offset, depth)

        # pose refinement
        for t in range(15):
            obj.setModelviewMatrix(rough_pred_pose)
            obj.findVisibleSamplePoint()

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

            # cropped image with initial pose as center
            crop_img = refine_frame[ey : ey + eh, ex : ex + ew].copy()
            if crop_img.shape[0] != boundingsize or crop_img.shape[1] != boundingsize:
                print("crop error!!")
                exit()

            if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
                print("no image")
                exit()

            # cv2.imshow("expand img", crop_img)
            # cropped mask for initial pose
            crop_mask = mask[ey : ey + eh, ex : ex + ew]
            # cropped edges for initial pose
            crop_edge = edge[ey : ey + eh, ex : ex + ew]
            # cv2.imshow("edge image", crop_edge)

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

        obj.setModelviewMatrix(pred_pose)
        obj.findVisibleSamplePoint()
        for p in obj.sharp_2d_pts:
            p = (int(p[0]), int(p[1]))
            demo = cv2.circle(demo, p, radius=2, color=(0, 0, 255), thickness=-1)
        cv2.imshow("demo", demo)
        cv2.waitKey(0)

        return pred_pose
    else:
        # can't detect the object
        print("can't detect object!!")
        return None
