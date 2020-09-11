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

from src.utils.utils import (
    load_classes,
    non_max_suppression,
    scale_coords,
    plot_one_box,
)
import tqdm
import cv2
from poseUtil import getPredictPose, getRotationError, ADD_error, ADDS_error

IMG_SIZE = 240
EXPAND_SIZE = 2.0
cfg = "cfg/yolov3-tiny3.cfg"
weights = "weights/best_yolo.pt"
conf_thres = 0.3
iou_thres = 0.2
device = "cuda"
obj_names = "data/wrs-ycb.names"
viewpt_class = 64
rot_class = 60
names = load_classes(obj_names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

# ignore warming
np.seterr(divide="ignore", invalid="ignore")


def get_centered_crop(topleft, botright):
    cropHeight = botright[1] - topleft[1]
    cropWidth = botright[0] - topleft[0]

    centerPoint = (topleft[0] + cropWidth / 2, topleft[1] + cropHeight / 2)

    cropSize = max(cropHeight, cropWidth)

    topleft_new = np.array(
        [centerPoint[0] - cropSize / 2, centerPoint[1] - cropSize / 2], dtype=int
    )
    botright_new = np.array(
        [centerPoint[0] + cropSize / 2, centerPoint[1] + cropSize / 2], dtype=int
    )

    return topleft_new, botright_new


def init():

    OM.setup(CFG.CAMERA_W, CFG.CAMERA_H)
    OM.setProjectMatrixWithIntr(CFG.CAMERA_MATRIX, CFG.CAMERA_W, CFG.CAMERA_H)

    obj = OM.ObjectModel()
    obj.loadObjectCADModel(CFG.CAD_MODEL)
    obj.setIntrinsicMatrix(CFG.CAMERA_MATRIX)

    obj.determineSharpEdges(0.05)
    obj.generateSamplePoints(0.001, 0.001)
    ###################### yolo ########################

    image_size = 416
    yolo_model = Darknet(cfg, image_size)  # default image size is 416
    # Load weights
    yolo_model.load_state_dict(torch.load(weights)["model"])
    # Eval mode
    yolo_model.to(device).eval()

    ################## rot model ################################

    rot_model = Magic_Net(viewpt_class=viewpt_class, rot_class=rot_class).cuda()
    rot_model = nn.DataParallel(rot_model)
    rot_model = torch.load(CFG.BEST_MODEL_ROT)
    rot_model.eval()

    ############### refine model ##########################
    refine_model = DeepIM().cuda()
    refine_model = nn.DataParallel(refine_model)
    refine_model = torch.load(CFG.BEST_MODEL_REFINE)
    refine_model.eval()

    return obj, yolo_model, rot_model, refine_model


def test(obj, yolo_model, rot_model, refine_model, image_file, pose_file):

    # read image
    img = cv2.imread(image_file)
    rot_frame = img.copy()
    refine_frame = img.copy()
    demo = img.copy()

    # resize image
    img = cv2.resize(img, (int(320), int(416)), interpolation=cv2.INTER_AREA)
    img = img[:, :, :3]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    # load image to the device
    img = torch.from_numpy(img).to(device)

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
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], demo.shape).round()

        for *xyxy, conf, cls in pred:
            label = "%s %.2f" % (names[int(cls)], conf)
            plot_one_box(xyxy, demo, label=label, color=colors[int(cls)])
            if names[int(cls)] == "009_gelatin_box":
                foundObject = True
                croptopleft = [
                    int(xyxy[0].cpu().detach().numpy()),
                    int(xyxy[1].cpu().detach().numpy()),
                ]
                croplowright = [
                    int(xyxy[2].cpu().detach().numpy()),
                    int(xyxy[3].cpu().detach().numpy()),
                ]
                break
    if foundObject:
        # rot classifier
        upperleft_rand, lowerright_rand = get_centered_crop(croptopleft, croplowright)

        if (
            int(upperleft_rand[1]) < 0
            or int(lowerright_rand[1]) >= demo.shape[0]
            or int(upperleft_rand[0]) < 0
            or int(lowerright_rand[0]) > demo.shape[1]
        ):
            return

        img_crop = rot_frame[
            int(upperleft_rand[1]) : int(lowerright_rand[1]),
            int(upperleft_rand[0]) : int(lowerright_rand[0]),
        ]

        l = int(lowerright_rand[0]) - int(upperleft_rand[0])

        img_crop = cv2.resize(img_crop, (IMG_SIZE, IMG_SIZE))
        crop_demo = img_crop.copy()

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
        # crop_demo = cv2.circle(
        #     crop_demo,
        #     (int(position[0, 0]), int(position[0, 1])),
        #     radius=5,
        #     color=(255, 0, 0),
        #     thickness=-1,
        # )
        # cv2.imshow("crop", crop_demo)
        offset = position[:, :2]
        offset = np.array(upperleft_rand) + offset.reshape(2) - principle_pt

        depth = 0.7
        rough_pred_pose = obj.label2pose(viewpt, rot, offset, depth)

        # pose refinement
        for t in range(1):
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
            print("crop image size ", crop_img.shape)

            if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
                return
            # cv2.imshow("expand img", crop_img)
            # cropped mask for initial pose
            crop_mask = mask[ey : ey + eh, ex : ex + ew]
            # cropped edges for initial pose
            crop_edge = edge[ey : ey + eh, ex : ex + ew]

            # apply rotation on the initial pose to move to it to the center
            rough_pose_at_center = obj.rotatePoseWithAngle(
                rough_pred_pose, horizontalR_ori, verticalR_ori
            )

            # calculate the resize scale
            rescaleValue = float(IMG_SIZE) / crop_img.shape[0]
            originalSize = crop_img.shape[0]

            # resize rgb image
            crop_img = cv2.resize(
                crop_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
            )

            crop_img = crop_img[:, :, :3].transpose(2, 0, 1)

            crop_img = Variable(torch.from_numpy(crop_img).cuda()).float() / 255.0
            crop_img = crop_img.unsqueeze(0)

            # load edge image
            edge_img = cv2.resize(
                crop_edge, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
            )
            edge_img = edge_img[:, :, np.newaxis].transpose(2, 0, 1)

            edge_img = Variable(torch.from_numpy(edge_img).cuda()).float() / 255.0
            edge_img = edge_img.unsqueeze(0)

            # load the mask image
            mask_img = cv2.resize(
                crop_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
            )
            mask_img = mask_img[:, :, np.newaxis].transpose(2, 0, 1)

            mask_img = Variable(torch.from_numpy(mask_img).cuda()).float() / 255.0
            mask_img = mask_img.unsqueeze(0)

            flow_inputData = torch.cat((mask_img, edge_img, crop_img), 1,)

            flow_input = Variable(flow_inputData)

            rot, trans, dist, opticalFlow, segmentMask = refine_model(flow_input)
            seg_pred = (
                torch.argmax(segmentMask, 1, keepdim=True)
                .float()
                .squeeze(1)
                .cpu()
                .detach()
                .numpy()
            )

            test_mask = cv2.resize(
                seg_pred[0], (originalSize, originalSize), interpolation=cv2.INTER_AREA,
            )

            mask[ey : ey + eh, ex : ex + ew] = test_mask

            cv2.imshow("pred_mask", mask)
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
                                float(crop_img.shape[-2]) / 2,
                            ],
                            [
                                0.0,
                                CFG.CAMERA_MATRIX[1, 1],
                                float(crop_img.shape[-2]) / 2,
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
            rough_pose_at_center = torch.from_numpy(rough_pose_at_center)
            rough_pose_at_center = rough_pose_at_center.unsqueeze(0)

            pred_pose = getPredictPose(
                rough_pose_at_center, rot, trans, dist, crop_img.shape[-2], rescaleValue
            )

            pred_pose = obj.rotatePoseWithAngle(
                pred_pose[0].cpu().detach().numpy(), -horizontalR_ori, -verticalR_ori
            )

            obj.setModelviewMatrix(pred_pose)
            obj.findVisibleSamplePoint()

            rough_pred_pose = pred_pose

            demotemp = demo.copy()

            # draw image
            for p in obj.sharp_2d_pts:
                p = (int(p[0]), int(p[1]))
                demotemp = cv2.circle(
                    demotemp, p, radius=2, color=(0, 0, 255), thickness=-1
                )

            cv2.imshow("frame", demotemp)
            cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    obj, yolo_model, rot_model, refine_model = init()
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

    index = 100
    image_names = image_names[index : index + 1]
    pose_names = pose_names[index : index + 1]

    datalist = list(zip(image_names, pose_names))

    for data in tqdm.tqdm(datalist):
        image_name, pose_name = data
        test(obj, yolo_model, rot_model, refine_model, image_name, pose_name)

    # test(yolo_model)
