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
from models.model import Magic_Net, Refine_Net, PSPNet

from src.utils.utils import (
    load_classes,
    non_max_suppression,
    scale_coords,
    plot_one_box,
)

import cv2

IMG_SIZE = 240
EXPAND_SIZE = 2.0

if __name__ == "__main__":

    OM.setup(CFG.CAMERA_W, CFG.CAMERA_H)
    OM.setProjectMatrixWithIntr(CFG.CAMERA_MATRIX, CFG.CAMERA_W, CFG.CAMERA_H)

    obj = OM.ObjectModel()
    obj.loadObjectCADModel(CFG.CAD_MODEL)
    obj.setIntrinsicMatrix(CFG.CAMERA_MATRIX)

    obj.determineSharpEdges(0.05)
    obj.generateSamplePoints(0.001, 0.001)

    ###################### yolo ########################
    webcam = "4"
    cfg = "cfg/yolov3-tiny3.cfg"
    weights = "best_model_yolo.pth"
    conf_thres = 0.3
    iou_thres = 0.2
    device = "cuda"
    obj_names = "data/wrs.names"
    image_size = 416

    names = load_classes(obj_names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    yolo_model = Darknet(cfg, image_size)  # default image size is 416

    # Load weights
    yolo_model.load_state_dict(torch.load(weights)["model"])
    # Eval mode
    yolo_model.to(device).eval()

    ################### magic net ########################
    viewpt_class = 64
    rot_class = 60

    rot_model = Magic_Net(viewpt_class=viewpt_class, rot_class=rot_class).cuda()
    rot_model = nn.DataParallel(rot_model)
    rot_model = torch.load("best_model_rot.pth")
    rot_model.eval()

    ################# refine net ###########################
    refine_model = Refine_Net().cuda()

    refine_model = nn.DataParallel(refine_model)
    refine_model = torch.load("best_model_refine.pth")
    refine_model.eval()

    ################ PSP net ##################################
    PSP_model = PSPNet().cuda()
    PSP_model = nn.DataParallel(PSP_model)
    PSP_model = torch.load("best_model_psp.pth")
    PSP_model.eval()

    softmax = nn.Softmax2d()

    # init camera
    cap = cv2.VideoCapture(4)
    cap.set(3, CFG.CAMERA_W)
    cap.set(4, CFG.CAMERA_H)

    if not cap.isOpened():
        print("can't open")
        exit()

    while True:
        # read image
        ret, frame = cap.read()
        rot_frame = frame.copy()
        refine_frame = frame.copy()
        demo = frame.copy()

        # resize image
        frame = cv2.resize(frame, (int(320), int(416)), interpolation=cv2.INTER_AREA)
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
        cropIndex = None

        foundObject = False
        # Process detections
        if pred is not None and len(pred):
            # Rescale boxes from img_size to demo size
            pred[:, :4] = scale_coords(frame.shape[2:], pred[:, :4], demo.shape).round()

            for *xyxy, conf, cls in pred:
                label = "%s %.2f" % (names[int(cls)], conf)
                plot_one_box(xyxy, demo, label=label, color=colors[int(cls)])
                if names[int(cls)] == "Large Bolt":
                    foundObject = True
                    cropIndex = [
                        int(xyxy[1].cpu().detach().numpy()),
                        int(xyxy[3].cpu().detach().numpy()),
                        int(xyxy[0].cpu().detach().numpy()),
                        int(xyxy[2].cpu().detach().numpy()),
                    ]
                    break

        if foundObject:
            # rot classifier
            l1 = cropIndex[1] - cropIndex[0]
            l2 = cropIndex[3] - cropIndex[2]
            l = max(l1, l2)
            img_crop = np.zeros((l, l, 3))

            img_crop = rot_frame[
                cropIndex[0] : cropIndex[0] + l, cropIndex[2] : cropIndex[2] + l
            ]

            if img_crop.shape[0] == 0 or img_crop.shape[1] == 0:
                continue

            img_crop = cv2.resize(img_crop, (IMG_SIZE, IMG_SIZE))
            # cv2.imshow("crop", img_crop)

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
            offset = (
                np.array([cropIndex[2], cropIndex[0]])
                + offset.reshape(2)
                - principle_pt
            )

            depth = 0.4
            rough_pred_pose = obj.label2pose(viewpt, rot, offset, depth)

            # pose refinement
            horizontalR_ori, verticalR_ori = obj.getCenterAngle(rough_pred_pose)

            obj.setModelviewMatrix(rough_pred_pose)
            obj.findVisibleSamplePoint()
            # draw init pose
            # for p in obj.sharp_2d_pts:
            #     demo = cv2.circle(
            #         demo,
            #         (int(p[0]), int(p[1])),
            #         radius=2,
            #         color=(0, 255, 0),
            #         thickness=-1,
            #     )

            # generate preprocessed data
            # inital pose mask
            mask = obj.getVisibleArea(refine_frame)

            # get edge img
            edge = obj.getEdge(refine_frame.shape[0], refine_frame.shape[1])

            # find the crop size
            [x, y, w, h] = cv2.boundingRect(mask)

            boundingsize = max(w, h) * EXPAND_SIZE

            ex = int(x + (w - boundingsize) / 2)
            ey = int(y + (h - boundingsize) / 2)
            ew = int(boundingsize)
            eh = int(boundingsize)

            # cropped image with initial pose as center
            crop_img = refine_frame[ey : ey + eh, ex : ex + ew]

            if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
                continue
            cv2.imshow("expand img", crop_img)

            # cropped mask for initial pose
            crop_mask = mask[ey : ey + eh, ex : ex + ew]
            # cropped edges for initial pose
            crop_edge = edge[ey : ey + eh, ex : ex + ew]

            # bounding box for target pose
            # crop_bounding = np.zeros((eh, ew), np.uint8)
            # crop_bounding[
            #     max(0, cropIndex[0] - ey) : min(eh, cropIndex[1] - ey),
            #     max(0, cropIndex[2] - ex) : min(ew, cropIndex[3] - ex),
            # ] = 255

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

            # ori_crop_image = crop_img.copy()
            # result_demo = np.zeros_like(crop_img)

            crop_img = crop_img[:, :, :3].transpose(2, 0, 1)

            crop_img = Variable(torch.from_numpy(crop_img).cuda()).float()
            crop_img = crop_img.unsqueeze(0)

            # psp net semantic segmentation
            predictMask = PSP_model(crop_img)

            predictMask = torch.argmax(predictMask, 1, keepdim=True).float()
            predictMask = (predictMask == 1).float().detach()

            # result_demo[pred] = ori_crop_image[pred]
            # result_demo = cv2.resize(result_demo, (800, 800), interpolation=cv2.INTER_AREA)
            # cv2.imshow("pred mask", result_demo)

            # load edge image
            edge_img = cv2.resize(
                crop_edge, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
            )
            edge_img = edge_img[:, :, np.newaxis].transpose(2, 0, 1)

            edge_img = Variable(torch.from_numpy(edge_img).cuda()).float()
            edge_img = edge_img.unsqueeze(0)

            # load bounding image
            # bounding_img = cv2.resize(
            #     crop_bounding, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
            # )
            # bounding_img = bounding_img[:, :, np.newaxis].transpose(2, 0, 1)

            # bounding_img = Variable(torch.from_numpy(bounding_img).cuda()).float()
            # bounding_img = bounding_img.unsqueeze(0)

            # load the mask image
            mask_img = cv2.resize(
                crop_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
            )
            mask_img = mask_img[:, :, np.newaxis].transpose(2, 0, 1)

            mask_img = Variable(torch.from_numpy(mask_img).cuda()).float()
            mask_img = mask_img.unsqueeze(0)

            # load init pose
            initPose = Variable(torch.from_numpy(rough_pose_at_center).cuda()).float()
            initPose = initPose.unsqueeze(0)

            inputData = torch.cat((edge_img, mask_img, predictMask, crop_img), 1)
            rot, trans, dist = refine_model(inputData)

            trans = trans.unsqueeze(1)
            trans = (trans - 0.5) * IMG_SIZE

            dist = dist.unsqueeze(1)

            # generate the rotation pose
            pred_rot_pose = torch.bmm(initPose, tgm.angle_axis_to_rotation_matrix(rot))

            # update the camera matrix because the input image is resize
            refine_camera_matrix = torch.tensor(
                [
                    [
                        [
                            CFG.CAMERA_MATRIX[0, 0] * rescaleValue,
                            0.0,
                            float(crop_img.shape[-2]) / 2,
                        ],
                        [
                            0.0,
                            CFG.CAMERA_MATRIX[1, 1] * rescaleValue,
                            float(crop_img.shape[-1]) / 2,
                        ],
                        [0.0, 0.0, 1.0],
                    ]
                ]
            ).cuda()

            dist_pose = pred_rot_pose.clone()
            dist_pose[:, 2, 3] = pred_rot_pose[:, 2, 3] / dist[:, 0, 0]

            horizontalR = torch.atan2(
                trans[:, :, 0],
                torch.tensor(CFG.CAMERA_MATRIX[0, 0] * rescaleValue).cuda(),
            )

            verticalR = -torch.atan2(
                trans[:, :, 1],
                torch.sqrt(
                    trans[:, :, 0] * trans[:, :, 0]
                    + torch.tensor(
                        CFG.CAMERA_MATRIX[1, 1]
                        * CFG.CAMERA_MATRIX[1, 1]
                        * rescaleValue
                        * rescaleValue
                    ).cuda()
                )
                * torch.tensor(
                    CFG.CAMERA_MATRIX[1, 1] / CFG.CAMERA_MATRIX[0, 0]
                ).cuda(),
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

            pred_pose = torch.bmm(rotation_matrix, dist_pose)

            # apply rotation on the initial pose to move to it to the center

            pred_pose = obj.rotatePoseWithAngle(
                pred_pose.cpu().detach().numpy(), -horizontalR_ori, -verticalR_ori
            )

            obj.setModelviewMatrix(pred_pose)
            obj.findVisibleSamplePoint()

            # draw image
            for p in obj.sharp_2d_pts:
                demo = cv2.circle(demo, p, radius=2, color=(0, 0, 255), thickness=-1)

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv2.imshow("frame", demo)
        if cv2.waitKey(100) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
