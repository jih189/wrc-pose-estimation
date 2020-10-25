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

from src.utils.utils import (
    load_classes,
    non_max_suppression,
    scale_coords,
    plot_one_box,
)
import tqdm
import cv2
from poseUtil import getPredictPose, getRotationError, ADD_error, ADDS_error

counter = Value("i", 0)

IMG_SIZE = 240
EXPAND_SIZE = 2.0
cfg = "cfg/yolov3-tiny3.cfg"
weights = "weights/best_yolo.pt"
conf_thres = 0.3
iou_thres = 0.2
obj_names = "data/wrs-ycb.names"
viewpt_class = CFG.VIEWPOINT_NUM
rot_class = 60
names = load_classes(obj_names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

# ignore warming
np.seterr(divide="ignore", invalid="ignore")

totalADD = []
totalADDS = []


def init():

    ###################### yolo ########################

    # image_size = 416
    # yolo_model = Darknet(cfg, image_size)  # default image size is 416
    # # Load weights
    # yolo_model.load_state_dict(torch.load(weights)["model"])
    # # Eval mode
    # yolo_model.eval()
    # yolo_model.share_memory()

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
    obj.generateSamplePoints(0.0001, 0.0001)
    return obj


def yolo_detect(yolo_model, img):
    orishape = img.shape
    # resize image
    img = cv2.resize(img, (int(320), int(416)), interpolation=cv2.INTER_AREA)
    img = img[:, :, :3]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    # load image to the device
    img = torch.from_numpy(img)

    # convert image to be used
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = yolo_model(img)[0].float()

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
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], orishape).round()

        for *xyxy, conf, cls in pred:
            # label = "%s %.2f" % (names[int(cls)], conf)
            # plot_one_box(xyxy, demo, label=label, color=colors[int(cls)])
            if names[int(cls)] == "pulley":
                # if names[int(cls)] == "Pulley":
                foundObject = True
                croptopleft = [
                    int(xyxy[0].detach().numpy()),
                    int(xyxy[1].detach().numpy()),
                ]
                croplowright = [
                    int(xyxy[2].detach().numpy()),
                    int(xyxy[3].detach().numpy()),
                ]
                break
    return foundObject, croptopleft, croplowright


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
    return True, crop_upperleft, crop_lowerright


def testData(obj, yolo_model, rot_model, refine_model, d):

    image_name, pose_name = d
    # read image
    img = cv2.imread(image_name)
    # cv2.imshow("origin image", img)
    rot_frame = img.copy()
    refine_frame = img.copy()
    demo = img.copy()

    # read the pose
    targetPose = np.load(pose_name)
    targetPose = OM.symmetricRemove_housing(targetPose)
    targetPose_temp = targetPose.copy()

    # use yolo to generate the bounding box
    # foundObject, croptopleft, croplowright = yolo_detect(yolo_model, img)

    # use obj model to generate the bounding box
    foundObject, croptopleft, croplowright = generateBoundingbox(obj, targetPose)

    targetPose = torch.from_numpy(targetPose)

    if foundObject:
        # rot classifier
        upperleft_rand, lowerright_rand = OM.get_centered_crop(
            croptopleft, croplowright
        )

        if (
            int(upperleft_rand[1]) < 0
            or int(lowerright_rand[1]) >= demo.shape[0]
            or int(upperleft_rand[0]) < 0
            or int(lowerright_rand[0]) > demo.shape[1]
        ):
            return

        # crop the image for rot classifier
        img_crop = rot_frame[
            int(upperleft_rand[1]) : int(lowerright_rand[1]),
            int(upperleft_rand[0]) : int(lowerright_rand[0]),
        ]

        # cv2.imshow("cropped image", img_crop)

        l = int(lowerright_rand[0]) - int(upperleft_rand[0])

        # resize the image so rot classifier can process
        img_crop = cv2.resize(img_crop, (IMG_SIZE, IMG_SIZE))

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
        position *= l
        offset = position[:, :2]
        offset = np.array(upperleft_rand) + offset.reshape(2) - principle_pt

        depth = 0.5

        # get the rough pose from view point, rotation, offset, and depth
        rough_pred_pose = obj.label2pose(viewpt, rot, offset, depth)

        # # pose refinement
        for t in range(5):
            obj.setModelviewMatrix(rough_pred_pose)
            obj.findVisibleSamplePoint()

            horizontalR_ori, verticalR_ori = obj.getCenterAngle(rough_pred_pose)

            # get init mask
            mask = obj.getVisibleArea()

            # get edge img
            edge = obj.getEdge(refine_frame.shape[0], refine_frame.shape[1])

            # find the crop size
            [x, y, w, h] = cv2.boundingRect(mask)

            boundingsize = max(w, h) * EXPAND_SIZE

            # get center point from pose
            centerPoint = obj.project3Dto2D((0, 0, 0), rough_pred_pose)

            ex = int(centerPoint[0] - boundingsize / 2)
            ey = int(centerPoint[1] - boundingsize / 2)
            ew = int(boundingsize)
            eh = int(boundingsize)

            # cropped image with initial pose as center
            crop_img = refine_frame[ey : ey + eh, ex : ex + ew].copy()
            if crop_img.shape[0] != boundingsize or crop_img.shape[1] != boundingsize:
                return

            if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
                return
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
            rescaleValue = float(IMG_SIZE) / crop_img.shape[0]

            # resize rgb image
            crop_img = cv2.resize(
                crop_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
            )

            crop_demo = crop_img.copy()
            invflow = np.zeros(crop_img.shape)

            crop_img = crop_img[:, :, :3].transpose(2, 0, 1)

            crop_img = Variable(torch.from_numpy(crop_img)).float() / 255.0
            crop_img = crop_img.unsqueeze(0)

            # load edge image
            edge_img = cv2.resize(
                crop_edge, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
            )
            edge_img = edge_img[:, :, np.newaxis].transpose(2, 0, 1)

            edge_img = Variable(torch.from_numpy(edge_img)).float() / 255.0
            edge_img = edge_img.unsqueeze(0)

            # load the mask image
            mask_img = cv2.resize(
                crop_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
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
            obj.setModelviewMatrix(pred_pose)
            obj.findVisibleSamplePoint()

            print(t, " refine")
            seg_pred = (
                torch.argmax(segmentMask, 1, keepdim=True)
                .float()
                .squeeze(1)
                .cpu()
                .detach()
                .numpy()
            )

            for y in range(invflow.shape[0]):
                for x in range(invflow.shape[1]):
                    if seg_pred[0, y, x] == 1:
                        invflow[y, x] = [0, 255, 0]
            for y in range(invflow.shape[0]):
                for x in range(invflow.shape[1]):
                    # if x % 10 == 0 and y % 10 == 0:
                    [mx, my] = opticalFlow[0, :2, y, x].cpu().detach().numpy()
                    if mx != 0.0 or my != 0.0:
                        mx = int((mx - 0.5) * IMG_SIZE)
                        my = int((my - 0.5) * IMG_SIZE)
                        if (
                            x + mx >= 0
                            and x + mx < crop_demo.shape[0]
                            and y + my >= 0
                            and y + my < crop_demo.shape[1]
                        ):
                            invflow = cv2.circle(
                                invflow,
                                (x + mx, y + my),
                                radius=0,
                                color=(255, 255, 255),
                                thickness=-1,
                            )

                            if x % 20 == 0 and y % 20 == 0:

                                crop_demo = cv2.line(
                                    crop_demo,
                                    (x, y),
                                    (x + mx, y + my),
                                    color=(0, 255, 0),
                                    thickness=1,
                                )
                                crop_demo = cv2.circle(
                                    crop_demo,
                                    (x + mx, y + my),
                                    radius=0,
                                    color=(255, 0, 0),
                                    thickness=-1,
                                )
            cv2.imshow("flow", crop_demo)

            cv2.imshow("mask", seg_pred[0])

            demo_temp = demo.copy()
            # draw image
            for p in obj.sharp_2d_pts:
                p = (int(p[0]), int(p[1]))
                demo_temp = cv2.circle(
                    demo_temp, p, radius=0, color=(0, 255, 0), thickness=-1
                )

            cv2.imshow("demo", demo_temp)
            cv2.waitKey(0)

            rough_pred_pose = pred_pose

            _, croptopleft, croplowright = generateBoundingbox(obj, pred_pose)

            upperleft_rand, lowerright_rand = OM.get_centered_crop(
                croptopleft, croplowright
            )


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

    indexnum = 100
    image_names = image_names[indexnum : indexnum + 1]
    pose_names = pose_names[indexnum : indexnum + 1]

    datalist = list(zip(image_names, pose_names))

    obj = obj_init()
    for d in tqdm.tqdm(datalist):
        testData(obj, yolo_model, rot_model, refine_model, d)

    # print("average add is ", sum(totalADD) / len(totalADD))
    # print("average adds is ", sum(totalADDS) / len(totalADDS))

