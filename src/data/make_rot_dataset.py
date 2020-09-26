# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import cv2
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

counter = Value("i", 0)
output_counter = Value("i", 0)
vparr = Array("i", [0] * 64)
rotarr = Array("i", [0] * 60)

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


def yoloDetection(yolo_model, img):
    # use yolo to detect object
    yolo_input = img.copy()
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
    croptopleft, croplowright = None, None

    foundObject = False
    # Process detections
    if yolo_pred is not None and len(yolo_pred):
        # Rescale boxes from img_size to demo size
        yolo_pred[:, :4] = scale_coords(
            yolo_input.shape[2:], yolo_pred[:, :4], img.shape
        ).round()

        for *xyxy, conf, cls in yolo_pred:
            if names[int(cls)] == CFG.OBJ_NAME:
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
    if not foundObject:
        return None
    return (croptopleft, croplowright)


def process_data(args):
    global counter
    global output_counter
    global vparr
    global rotarr

    output_filepath = CFG.PROCESSED_DATA_PATH

    obj = init()

    # parse input
    (id, datalist, yolo_model) = args
    useYolo = False if yolo_model == None else True
    (img_names, pose_names) = list(zip(*datalist))

    while True:
        with counter.get_lock():
            current_index = counter.value
            counter.value += 1

        if current_index >= len(datalist):
            return

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

        try:
            # read image and pose
            img = cv2.imread(img_names[current_index])
            img = np.array(img)

            pose = np.load(pose_names[current_index])
        except Exception as e:
            print("error in load data!!!!")
            print(str(e))

        depth = pose[2, 3]

        # generate the real bounding box for object
        obj.setModelviewMatrix(pose)

        if useYolo:
            # run yolo on image
            try:
                yoloresult = yoloDetection(yolo_model, img)
                if yoloresult == None:
                    continue

            except Exception as e:
                print("ERROR: yolo detection error")
                print(str(e))
                return

            upperleft, lowerright = yoloresult
            # we need to ensure the center point of the object is in the bounding box
            center_point = obj.project3Dto2D((0.0, 0.0, 0.0), pose)

            if (
                center_point[0] < upperleft[0]
                or center_point[0] >= lowerright[0]
                or center_point[1] < upperleft[1]
                or center_point[1] >= lowerright[1]
            ):
                continue
        else:
            try:
                # random generate a bounding box around the object with given pose
                upperleft, lowerright = obj.findVisibleSamplePoint()
                upperleft, lowerright = (
                    np.array(upperleft).reshape(2),
                    np.array(lowerright).reshape(2),
                )

                high = np.clip(10 / depth, 0, 50)
                upperleft = (
                    upperleft - np.random.uniform(0, high, upperleft.shape)
                ).astype(np.int)
                lowerright = (
                    lowerright + np.random.uniform(0, high, lowerright.shape)
                ).astype(np.int)
            except Exception as e:
                print("error in generate bounding box!!!!")
                print(upperleft)
                print(str(e))

        crop_upperleft, crop_lowerright = get_centered_crop(upperleft, lowerright)

        # for making life easier, we drop the image if the crop bounding box is out of image
        if (
            int(crop_upperleft[1]) < 0
            or int(crop_lowerright[1]) >= img.shape[0]
            or int(crop_upperleft[0]) < 0
            or int(crop_lowerright[0]) >= img.shape[1]
        ):
            continue

        # get view point, inplance rotation, offset from center, and depth from the pose
        viewPoint, inplaneRotation, offsetFromCenter, depth = obj.getLabel()
        inplaneRotation = inplaneRotation % (2 * np.pi) / (2 * np.pi / 60)
        if np.isnan(inplaneRotation):
            # the inplane rotation is invalid when y axis is pointing to camera
            continue

        try:
            cropImg = img[
                int(crop_upperleft[1]) : int(crop_lowerright[1]),
                int(crop_upperleft[0]) : int(crop_lowerright[0]),
            ]
        except Exception as e:
            print("error in cropping image!!!")
            print(str(e))

        with output_counter.get_lock():
            current_output_index = output_counter.value
            output_counter.value += 1

        np.save(
            output_filepath
            + "bounding"
            + "{:06d}".format(current_output_index)
            + ".npy",
            np.array(
                [
                    int(crop_upperleft[0]),
                    int(crop_upperleft[1]),
                    int(crop_lowerright[0]),
                    int(crop_lowerright[1]),
                ]
            ),
        )

        np.save(
            output_filepath + "{:06d}".format(current_output_index) + ".npy", pose,
        )

        cv2.imwrite(
            output_filepath + "crop" + "{:06d}".format(current_output_index) + ".png",
            cropImg,
        )

        cv2.imwrite(
            output_filepath + "{:06d}".format(current_output_index) + ".png", img,
        )

        inplaneRotation = int(inplaneRotation)
        vpidx = OM.cal_idx(viewPoint)

        with vparr.get_lock():
            vparr[vpidx] += 1

        with rotarr.get_lock():
            rotarr[inplaneRotation] += 1


@click.command()
@click.argument(
    "input_filepath", default=CFG.VERIFY_IMAGE_PATH, type=click.Path(exists=True)
)
@click.argument("output_filepath", default=CFG.PROCESSED_DATA_PATH, type=click.Path())
def main(input_filepath, output_filepath):
    global output_counter
    global vparr
    global rotarr
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    logger.info(f"Input directory: {input_filepath}")
    logger.info(f"Output directory: {output_filepath}")

    #################### yolo #########################
    # if you do not want to use yolo, comment this part
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
    ################################################

    # read images and poses
    input_path = Path(input_filepath)
    image_names, pose_names = [], []
    for f in input_path.iterdir():
        if f.match("*.png"):
            image_names.append(str(f))
        if f.match("*.npy"):
            pose_names.append(str(f))
    image_names.sort()
    pose_names.sort()

    # image_names = image_names[:1]
    # pose_names = pose_names[:1]

    # generate input for function
    datalist = list(zip(image_names, pose_names))

    inputP = []
    # if you do not want to use yolo, use pass None instead of model
    for o in range(cpu_count()):
        inputP.append((o, list(datalist), None))

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

    # calculate the weight of the dataset
    vp_weight_balance = []
    for i in range(64):
        if vparr[i] != 0:
            vp_weight_balance.append(1.0 / vparr[i])
        else:
            vp_weight_balance.append(1.0)

    np.save(
        output_filepath + "vp_weight.npy", vp_weight_balance,
    )

    rot_weight_balance = []
    for i in range(60):
        if rotarr[i] != 0:
            rot_weight_balance.append(1.0 / rotarr[i])
        else:
            rot_weight_balance.append(1.0)

    np.save(
        output_filepath + "rot_weight.npy", rot_weight_balance,
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
