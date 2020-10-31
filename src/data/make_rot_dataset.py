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
from scipy.spatial.transform import Rotation as R
from ctypes import c_bool

import torch
import torch.nn as nn

torch.multiprocessing.set_sharing_strategy("file_system")

# global variable for multiprocessing
counter = Value("i", 0)
output_counter = Value("i", 0)
vparr = Array("i", [0] * CFG.VIEWPOINT_NUM)
rotarr = Array("i", [0] * 60)
testTrigger = Value(c_bool, False)

# object init
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


# parallel function for process data
def process_data(args):
    global counter
    global output_counter
    global vparr
    global rotarr
    global testTrigger

    output_filepath = CFG.PROCESSED_DATA_PATH

    obj = init()

    # parse input
    (id, datalist) = args
    (img_names, pose_names) = list(zip(*datalist))

    while True:
        if testTrigger.value == True:
            break
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
            testTrigger.value = True

        center_pt = obj.project3Dto2D((0, 0, 0), pose)
        center_pt = (int(center_pt[0]), int(center_pt[1]))
        depth = pose[2, 3]

        visibleArea = -1

        # generate different rotation to increase the number of data
        for r in range(8):
            inplaneRotate = r * 45.0
            try:
                rot_img = OM.rotate_image(img, inplaneRotate, center_pt)
                rot_pose = OM.rotateAngle(pose, inplaneRotate)

                # generate the real bounding box for object
                obj.setModelviewMatrix(rot_pose)

                # random generate a bounding box around the object with given pose
                obj.findVisibleSamplePoint()

                # ensure the object is not outside of the view
                visiblemask = obj.getVisibleArea()
                if visibleArea == -1:
                    visibleArea = cv2.countNonZero(visiblemask)
                elif visibleArea * 0.95 > cv2.countNonZero(visiblemask):
                    continue

                # extract the bounding box
                bx, by, bw, bh = cv2.boundingRect(visiblemask)
                upperleft = np.array([bx, by])
                lowerright = np.array([bx + bw, by + bh])

                high = np.clip(10 / depth, 0, 50)
                upperleft = (
                    upperleft - np.random.uniform(0, high, upperleft.shape)
                ).astype(np.int)
                lowerright = (
                    lowerright + np.random.uniform(0, high, lowerright.shape)
                ).astype(np.int)
            except Exception as e:
                print("error in generate bounding box!!!!")
                print(str(e))
                testTrigger.value = True

            # adjust the bounding box
            crop_upperleft, crop_lowerright = OM.get_centered_crop(
                upperleft, lowerright
            )

            try:
                cropImg = np.zeros(
                    (
                        crop_lowerright[1] - crop_upperleft[1],
                        crop_lowerright[0] - crop_upperleft[0],
                        3,
                    ),
                    np.uint8,
                )
                upperleft_crop_inner = [
                    max(0, crop_upperleft[0]),
                    max(0, crop_upperleft[1]),
                ]
                lowerright_crop_inner = [
                    min(rot_img.shape[1], crop_lowerright[0]),
                    min(rot_img.shape[0], crop_lowerright[1]),
                ]
                cropImg[
                    upperleft_crop_inner[1]
                    - crop_upperleft[1] : lowerright_crop_inner[1]
                    - crop_upperleft[1],
                    upperleft_crop_inner[0]
                    - crop_upperleft[0] : lowerright_crop_inner[0]
                    - crop_upperleft[0],
                ] = rot_img[
                    int(upperleft_crop_inner[1]) : int(lowerright_crop_inner[1]),
                    int(upperleft_crop_inner[0]) : int(lowerright_crop_inner[0]),
                ]

                # get view point, inplance rotation, offset from center, and depth from the pose
                viewPoint, inplaneRotation, offsetFromCenter, depth = obj.getLabel()
                inplaneRotation = inplaneRotation % (2 * np.pi) / (np.pi / 30)
                if np.isnan(inplaneRotation):
                    # the inplane rotation is invalid when y axis is pointing to camera
                    continue
            except Exception as e:
                print("error in cropping image!!!")
                print(str(e))
                testTrigger.value = True
                cv2.waitKey(0)

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
                output_filepath + "{:06d}".format(current_output_index) + ".npy",
                rot_pose,
            )

            cv2.imwrite(
                output_filepath
                + "crop"
                + "{:06d}".format(current_output_index)
                + ".png",
                cropImg,
            )

            cv2.imwrite(
                output_filepath + "{:06d}".format(current_output_index) + ".png",
                rot_img,
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

    for o in range(cpu_count()):
        inputP.append((o, list(datalist)))

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
    for i in range(CFG.VIEWPOINT_NUM):
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
