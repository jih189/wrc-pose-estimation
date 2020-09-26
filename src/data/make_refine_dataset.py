# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import cv2
from tqdm import tqdm
import random
import src.common.object_model as OM
import src.configuration as CFG
from torch.multiprocessing import Pool, Value, cpu_count, Array

import torch
import torch.nn as nn

from models.models import Darknet  # set ONNX_EXPORT in models.py

from src.utils.utils import (
    non_max_suppression,
    load_classes,
    scale_coords,
    plot_one_box,
)

torch.multiprocessing.set_sharing_strategy("file_system")

EXPAND_SIZE = 2.0
RANDOM_NUM = 4

counter = Value("i", 0)
output_counter = Value("i", 0)

conf_thres = 0.35
iou_thres = 0.5
obj_names = "data/wrs-ycb.names"
names = load_classes(obj_names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


def init():
    # load the object mesh
    OM.setup(CFG.CAMERA_W, CFG.CAMERA_H)
    OM.setProjectMatrixWithIntr(CFG.CAMERA_MATRIX, CFG.CAMERA_W, CFG.CAMERA_H)

    obj = OM.ObjectModel()
    obj.setIntrinsicMatrix(CFG.CAMERA_MATRIX)
    obj.loadObjectCADModel(CFG.CAD_MODEL)

    obj.determineSharpEdges(0.8)
    obj.generateSamplePoints(0.001, 0.0001)
    return obj


def yoloDetection(yolo_model, img):
    # use yolo to detect object
    yolo_input = img.copy()
    demo = img.copy()
    # resize image
    yolo_input = cv2.resize(
        yolo_input, (int(320), int(416)), interpolation=cv2.INTER_AREA
    )
    yolo_input = yolo_input[:, :, :3]
    yolo_input = yolo_input[:, :, ::-1].transpose(2, 0, 1)
    yolo_input = np.ascontiguousarray(yolo_input)
    # load image to the device
    yolo_input = torch.from_numpy(yolo_input).float()

    # convert image to be used
    yolo_input /= 255.0  # 0 - 255 to 0.0 - 1.0
    if yolo_input.ndimension() == 3:
        yolo_input = yolo_input.unsqueeze(0)

    # Inference
    yolo_pred = yolo_model(yolo_input)[0].float()

    # Apply NMS
    yolo_pred = non_max_suppression(
        yolo_pred, conf_thres, iou_thres, classes=None, agnostic=False
    )

    yolo_pred = yolo_pred[0]
    # croptopleft, croplowright = None, None

    foundObject = False
    # Process detections
    if yolo_pred is not None and len(yolo_pred):
        # Rescale boxes from img_size to demo size
        yolo_pred[:, :4] = scale_coords(
            yolo_input.shape[2:], yolo_pred[:, :4], img.shape
        ).round()

        for *xyxy, conf, cls in yolo_pred:
            # label = "%s %.2f" % (names[int(cls)], conf)
            # plot_one_box(xyxy, demo, label=label, color=colors[int(cls)])
            if names[int(cls)] == CFG.OBJ_NAME:
                foundObject = True
                # croptopleft = [
                #     int(xyxy[0].cpu().detach().numpy()),
                #     int(xyxy[1].cpu().detach().numpy()),
                # ]
                # croplowright = [
                #     int(xyxy[2].cpu().detach().numpy()),
                #     int(xyxy[3].cpu().detach().numpy()),
                # ]

    # cv2.imshow("yolo", demo)
    # cv2.waitKey(0)
    # return (croptopleft, croplowright)
    return foundObject


def process_data(args):
    global counter
    global output_counter

    output_filepath = CFG.REFINE_DATA_PATH

    obj = init()

    # parse input
    (id, datalist, isYCB, yolo_model) = args

    useYolo = False if yolo_model == None else True

    if isYCB:
        (img_names, pose_names, mask_names) = list(zip(*datalist))
    else:
        (img_names, pose_names) = list(zip(*datalist))

    while True:
        with counter.get_lock():
            current_index = counter.value
            counter.value += 1

        if current_index >= len(datalist):
            return
        try:
            np.random.seed(current_index)

            # update the progress bar
            progress = int(50.0 * current_index / len(datalist))
            rest_progress = 50 - progress
            print(
                "Progress: ["
                + "=" * progress
                + " " * rest_progress
                + "]"
                + str(100.0 * current_index / len(datalist))
                + "%",
                end="\r",
                flush=True,
            )

            # read image and pose
            img = cv2.imread(img_names[current_index])
            pose = np.load(pose_names[current_index])

            # if this is symmetric object, you need to remove the redundent poses
            # pose = OM.symmetricRemove_housing(pose)
            # pose = OM.symmetricRemove(pose)

            if useYolo:
                # if yolo can't detect the object in the image, then ignore it.
                try:
                    yoloresult = yoloDetection(yolo_model, img)
                    if yoloresult == False:
                        continue

                except Exception as e:
                    print("ERROR: yolo detection error")
                    print(str(e))
                    return

            # generate set of random poses
            random_poses = obj.resample(pose, RANDOM_NUM)

            # generate the real bounding box for object
            obj.setModelviewMatrix(pose)
            obj.findVisibleSamplePoint()

            # # test
            # testimg = img.copy()
            # for p in obj.sharp_2d_pts:
            #     testimg = cv2.circle(
            #         testimg,
            #         (int(p[0]), int(p[1])),
            #         radius=1,
            #         color=(0, 255, 0),
            #         thickness=-1,
            #     )
            # cv2.imshow("test", testimg)
            # cv2.waitKey(0)

            if isYCB:
                # read the mask
                target_mask = cv2.imread(mask_names[current_index])
            else:
                # generate the mask from the pose
                target_mask = obj.getVisibleArea()

            # use random sample poses as the init pose
            for random_pose in random_poses:

                # get current pose if it is moved to the center
                horizontalR, verticalR = obj.getCenterAngle(random_pose)

                # set pose on object
                obj.setModelviewMatrix(random_pose)

                # generate edge of on the object
                obj.findVisibleSamplePoint()

                # generate preprocessed data
                # inital pose mask
                mask = obj.getVisibleArea()

                # get edge img
                edge = obj.getEdge(img.shape[0], img.shape[1])

                # find the crop size
                [_, _, w, h] = cv2.boundingRect(mask)

                boundingsize = max(w, h) * EXPAND_SIZE

                # get center point from pose
                centerPoint = obj.project3Dto2D((0, 0, 0), random_pose)

                ex = int(centerPoint[0] - boundingsize / 2)
                ey = int(centerPoint[1] - boundingsize / 2)
                ew = int(boundingsize)
                eh = int(boundingsize)

                if ew == 0 or eh == 0:
                    continue

                if (
                    ex < 0
                    or ey < 0
                    or ex + ew >= img.shape[1]
                    or ey + eh >= img.shape[0]
                ):
                    continue

                # generate opt flow
                if isYCB:
                    flowImg = obj.getOptFlowWithPosesAndMask(
                        boundingsize, boundingsize, pose, target_mask
                    )
                else:
                    flowImg = obj.getOptFlowWithPoses(boundingsize, boundingsize, pose)

                crop_flowImg = flowImg[ey : ey + eh, ex : ex + ew]

                # generate 3d image
                imgFor3d = obj.get3dimage(boundingsize, boundingsize)
                crop_3dImg = imgFor3d[ey : ey + eh, ex : ex + ew]

                # crop_img = img[ey : ey + eh, ex : ex + ew].copy()
                # cropped image with initial pose as center
                crop_img = img[ey : ey + eh, ex : ex + ew]
                # cropped mask for initial pose
                crop_mask = mask[ey : ey + eh, ex : ex + ew]
                # cropped edges for initial pose
                crop_edge = edge[ey : ey + eh, ex : ex + ew]

                # target mask
                crop_label_mask = target_mask[ey : ey + eh, ex : ex + ew]

                # apply rotation on the initial pose to move to it to the center
                current_pose_at_center = obj.rotatePoseWithAngle(
                    random_pose, horizontalR, verticalR
                )

                # apply same rotation as above
                target_pose_at_center = obj.rotatePoseWithAngle(
                    pose, horizontalR, verticalR
                )

                # get the 3d pts from init pose
                obj.setModelviewMatrix(current_pose_at_center)
                # generate edge of on the object
                obj.findVisibleSamplePoint()

                init3dpts = np.array(obj.visible_sharpedge_samplepoint)
                if init3dpts.shape[0] < 100:
                    print("no enough sample point in this pose! continue...")
                    print("with only ", init3dpts.shape[0], "  points")
                    continue

                # get the 3d sample pts from target pose
                obj.setModelviewMatrix(target_pose_at_center)
                # generate edge of on the object
                obj.findVisibleSamplePoint()

                target3dpts = np.array(obj.visible_sharpedge_samplepoint)
                if target3dpts.shape[0] < 100:
                    print("no enough sample point in this pose! continue...")
                    continue

                # print("write index ", current_index)

                with output_counter.get_lock():
                    current_output_index = output_counter.value
                    output_counter.value += 1

                cv2.imwrite(
                    output_filepath + "{:06d}".format(current_output_index) + "img.png",
                    crop_img,
                )

                cv2.imwrite(
                    output_filepath
                    + "{:06d}".format(current_output_index)
                    + "flow.png",
                    crop_flowImg,
                )

                cv2.imwrite(
                    output_filepath + "{:06d}".format(current_output_index) + "-3d.png",
                    crop_3dImg,
                )

                cv2.imwrite(
                    output_filepath
                    + "{:06d}".format(current_output_index)
                    + "mask.png",
                    crop_mask,
                )
                cv2.imwrite(
                    output_filepath
                    + "{:06d}".format(current_output_index)
                    + "edge.png",
                    crop_edge,
                )
                cv2.imwrite(
                    output_filepath
                    + "{:06d}".format(current_output_index)
                    + "labelmask.png",
                    crop_label_mask,
                )

                np.save(
                    output_filepath
                    + "{:06d}".format(current_output_index)
                    + "initPose.npy",
                    current_pose_at_center,
                )
                np.save(
                    output_filepath
                    + "{:06d}".format(current_output_index)
                    + "targetPose.npy",
                    target_pose_at_center,
                )
                np.save(
                    output_filepath
                    + "{:06d}".format(current_output_index)
                    + "init3dPt.npy",
                    init3dpts,
                )
                np.save(
                    output_filepath
                    + "{:06d}".format(current_output_index)
                    + "target3dPt.npy",
                    target3dpts,
                )
        except Exception as e:
            print(str(e))


@click.command()
@click.argument(
    "input_filepath", default=CFG.VERIFY_IMAGE_PATH, type=click.Path(exists=True)
)
@click.argument("output_filepath", default=CFG.REFINE_DATA_PATH, type=click.Path())
def main(input_filepath, output_filepath):
    global output_counter
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    logger.info(f"Input directory: {input_filepath}")
    logger.info(f"Output directory: {output_filepath}")

    # #################### yolo #########################
    # weights = "weights/best_yolo.pt"
    # cfg = "cfg/yolov3-tiny3.cfg"
    # image_size = 416
    # yolo_model = Darknet(cfg, image_size)  # default image size is 416
    # # Load weights
    # yolo_model.load_state_dict(torch.load(weights)["model"])
    # # Eval mode
    # yolo_model.eval()
    # # make the model can be shared by multiple processes
    # yolo_model.share_memory()

    # read images and poses
    input_path = Path(input_filepath)
    image_names, pose_names, mask_names = [], [], []
    for f in input_path.iterdir():
        if f.match("*.png"):
            image_names.append(str(f))
        if f.match("*.npy"):
            pose_names.append(str(f))
    image_names.sort()
    pose_names.sort()

    # load the label mask
    mask_file_inputpath = input_filepath + "label_mask/"
    mask_file_path = Path(mask_file_inputpath)
    isYCB = False
    if mask_file_path.exists() and mask_file_path.is_dir():
        print("Dataset is YCB dataset")
        isYCB = True
        for f in mask_file_path.iterdir():
            if f.match("*-label.png"):
                mask_names.append(str(f))
        mask_names.sort()

    # image_names = image_names[:5]
    # pose_names = pose_names[:5]

    # generate input for function
    if isYCB:
        datalist = list(zip(image_names, pose_names, mask_names))
    else:
        datalist = list(zip(image_names, pose_names))

    inputP = []
    for o in range(cpu_count()):
        inputP.append((o, list(datalist), isYCB, None))

    with Pool() as p:
        p.imap_unordered(process_data, inputP)
        p.close()
        p.join()

    current_index = output_counter.value

    print("")
    logger.info(f"Number of images generated = {current_index}")

    # update the train.txt and val.txt output_filepath
    val_list = random.sample(range(current_index), int((current_index + 1) * 0.3))
    train_list = [i for i in range(current_index) if i not in val_list]

    f = open(output_filepath + "train.txt", "w")
    for i in train_list:
        f.write("{:06d}".format(i) + "\n")
    f.close()

    f = open(output_filepath + "val.txt", "w")
    for i in val_list:
        f.write("{:06d}".format(i) + "\n")
    f.close()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
