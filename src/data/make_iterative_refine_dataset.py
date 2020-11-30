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
from scipy.spatial.transform import Rotation as R
from ctypes import c_bool

import torch
import torch.nn as nn

from src.utils.utils import (
    non_max_suppression,
    load_classes,
    scale_coords,
    plot_one_box,
)

torch.multiprocessing.set_sharing_strategy("file_system")

counter = Value("i", 0)
output_counter = Value("i", 0)
testTrigger = Value(c_bool, False)


def init():
    # load the object mesh
    OM.setup(CFG.CAMERA_W, CFG.CAMERA_H)
    OM.setProjectMatrixWithIntr(CFG.CAMERA_MATRIX, CFG.CAMERA_W, CFG.CAMERA_H)

    obj = OM.ObjectModel()
    obj.setIntrinsicMatrix(CFG.CAMERA_MATRIX)
    obj.loadObjectCADModel(CFG.CAD_MODEL)

    obj.determineSharpEdges(0.8)
    obj.generateSamplePoints(0.0001)
    return obj


def process_data(args):
    global counter
    global output_counter
    global testTrigger

    output_filepath = CFG.REFINE_ITERATIVE_DATA_PATH

    obj = init()

    # parse input
    (id, datalist, isYCB) = args
    (img_names, pose_names) = list(zip(*datalist))

    while True:
        if testTrigger.value == True:
            break
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

            center_pt = obj.project3Dto2D((0, 0, 0), pose)
            center_pt = (int(center_pt[0]), int(center_pt[1]))

            visibleArea = -1

            # rotate the plane so it can increase the number of data
            for r in range(8):
                inplaneRotate = r * 45.0

                rot_img = OM.rotate_image(img, inplaneRotate, center_pt)
                rot_pose = OM.rotateAngle(pose, inplaneRotate)

                # generate set of random poses
                # random_pose = obj.resample(rot_pose, 1)[0]
                random_pose = obj.resamplePose(rot_pose, 0.01, 0.05, 0.08)
                # random_pose = rot_pose

                # generate the ground true mask for object
                obj.setModelviewMatrix(rot_pose)
                obj.findVisibleSamplePoint()

                if isYCB:
                    # read the mask
                    target_mask = cv2.imread(mask_names[current_index])
                else:
                    # generate the mask from the pose
                    target_mask = obj.getVisibleArea()

                if cv2.countNonZero(target_mask) == 0:
                    continue

                # ensure the object is not outside of the view
                if visibleArea == -1:
                    visibleArea = cv2.countNonZero(target_mask)
                elif visibleArea * 0.95 > cv2.countNonZero(target_mask):
                    continue

                # set pose on object
                obj.setModelviewMatrix(random_pose)

                # generate edge of on the object
                obj.findVisibleSamplePoint()

                # get edge img
                imgWithEdge = rot_img.copy()
                for p in obj.sharp_2d_pts:
                    p = (int(p[0]), int(p[1]))
                    imgWithEdge = cv2.circle(
                        imgWithEdge, p, radius=1, color=(0, 0, 255), thickness=-1
                    )

                with output_counter.get_lock():
                    current_output_index = output_counter.value
                    output_counter.value += 1

                cv2.imwrite(
                    output_filepath + "{:06d}".format(current_output_index) + "img.png",
                    rot_img,
                )

                cv2.imwrite(
                    output_filepath
                    + "{:06d}".format(current_output_index)
                    + "demo.png",
                    imgWithEdge,
                )
                cv2.imwrite(
                    output_filepath
                    + "{:06d}".format(current_output_index)
                    + "labelmask.png",
                    target_mask,
                )
                np.save(
                    output_filepath
                    + "{:06d}".format(current_output_index)
                    + "initPose.npy",
                    random_pose,
                )
                np.save(
                    output_filepath
                    + "{:06d}".format(current_output_index)
                    + "targetPose.npy",
                    rot_pose,
                )
        except Exception as e:
            print(str(e))


@click.command()
@click.argument(
    "input_filepath", default=CFG.VERIFY_IMAGE_PATH, type=click.Path(exists=True)
)
@click.argument(
    "output_filepath", default=CFG.REFINE_ITERATIVE_DATA_PATH, type=click.Path()
)
def main(input_filepath, output_filepath):
    global output_counter
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    logger.info(f"Input directory: {input_filepath}")
    logger.info(f"Output directory: {output_filepath}")

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
        inputP.append((o, list(datalist), isYCB))

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
