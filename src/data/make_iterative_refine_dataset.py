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
from torch.multiprocessing import (
    Pool,
    Value,
    cpu_count,
    Array,
    Process,
    set_start_method,
)
from scipy.spatial.transform import Rotation as R
from ctypes import c_bool

import torch
import torch.nn as nn

from models.model import Magic_Net

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


def init_pose_generation(obj, upperleft, lowerright, rot_model, img):
    # adjust the bounding box
    crop_upperleft, crop_lowerright = OM.get_centered_crop(upperleft, lowerright)
    crop_width = int(lowerright[0]) - int(upperleft[0])

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
        min(img.shape[1], crop_lowerright[0]),
        min(img.shape[0], crop_lowerright[1]),
    ]
    cropImg[
        upperleft_crop_inner[1]
        - crop_upperleft[1] : lowerright_crop_inner[1]
        - crop_upperleft[1],
        upperleft_crop_inner[0]
        - crop_upperleft[0] : lowerright_crop_inner[0]
        - crop_upperleft[0],
    ] = img[
        int(upperleft_crop_inner[1]) : int(lowerright_crop_inner[1]),
        int(upperleft_crop_inner[0]) : int(lowerright_crop_inner[0]),
    ]
    img_input = cv2.resize(
        cropImg, (CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=cv2.INTER_AREA
    )
    img_input = img_input[:, :, :3].transpose(2, 0, 1)
    img_input = img_input[np.newaxis, ...]
    img_input = torch.from_numpy(img_input)

    img_input = img_input.float()

    output = rot_model(img_input)

    c0 = torch.sigmoid(
        output[
            :,
            CFG.VIEWPOINT_NUM
            + CFG.ROTATION_NUM
            + 2 : CFG.VIEWPOINT_NUM
            + CFG.ROTATION_NUM
            + 4,
        ]
    ).data.numpy()
    c1 = torch.sigmoid(
        output[
            :,
            CFG.VIEWPOINT_NUM
            + CFG.ROTATION_NUM
            + 4 : CFG.VIEWPOINT_NUM
            + CFG.ROTATION_NUM
            + 6,
        ]
    ).data.numpy()
    c2 = torch.sigmoid(
        output[
            :,
            CFG.VIEWPOINT_NUM
            + CFG.ROTATION_NUM
            + 6 : CFG.VIEWPOINT_NUM
            + CFG.ROTATION_NUM
            + 8,
        ]
    ).data.numpy()
    c3 = torch.sigmoid(
        output[
            :,
            CFG.VIEWPOINT_NUM
            + CFG.ROTATION_NUM
            + 8 : CFG.VIEWPOINT_NUM
            + CFG.ROTATION_NUM
            + 10,
        ]
    ).data.numpy()
    c4 = torch.sigmoid(
        output[
            :,
            CFG.VIEWPOINT_NUM
            + CFG.ROTATION_NUM
            + 10 : CFG.VIEWPOINT_NUM
            + CFG.ROTATION_NUM
            + 12,
        ]
    ).data.numpy()
    c5 = torch.sigmoid(
        output[
            :,
            CFG.VIEWPOINT_NUM
            + CFG.ROTATION_NUM
            + 12 : CFG.VIEWPOINT_NUM
            + CFG.ROTATION_NUM
            + 14,
        ]
    ).data.numpy()
    c6 = torch.sigmoid(
        output[
            :,
            CFG.VIEWPOINT_NUM
            + CFG.ROTATION_NUM
            + 14 : CFG.VIEWPOINT_NUM
            + CFG.ROTATION_NUM
            + 16,
        ]
    ).data.numpy()
    c7 = torch.sigmoid(
        output[
            :,
            CFG.VIEWPOINT_NUM
            + CFG.ROTATION_NUM
            + 16 : CFG.VIEWPOINT_NUM
            + CFG.ROTATION_NUM
            + 18,
        ]
    ).data.numpy()

    c0 *= crop_width
    c1 *= crop_width
    c2 *= crop_width
    c3 *= crop_width
    c4 *= crop_width
    c5 *= crop_width
    c6 *= crop_width
    c7 *= crop_width

    c0 = np.array([upperleft[0], upperleft[1]]) + c0.reshape(2)
    c1 = np.array([upperleft[0], upperleft[1]]) + c1.reshape(2)
    c2 = np.array([upperleft[0], upperleft[1]]) + c2.reshape(2)
    c3 = np.array([upperleft[0], upperleft[1]]) + c3.reshape(2)
    c4 = np.array([upperleft[0], upperleft[1]]) + c4.reshape(2)
    c5 = np.array([upperleft[0], upperleft[1]]) + c5.reshape(2)
    c6 = np.array([upperleft[0], upperleft[1]]) + c6.reshape(2)
    c7 = np.array([upperleft[0], upperleft[1]]) + c7.reshape(2)

    _, rvec, tvec, _ = cv2.solvePnPRansac(
        np.array(obj.cornerPoints),
        np.array([c0, c1, c2, c3, c4, c5, c6, c7]),
        CFG.CAMERA_MATRIX,
        np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_EPNP,
    )
    rotMat, _ = cv2.Rodrigues(rvec)
    pose = np.identity(4)
    pose[:3, :3] = rotMat
    pose[0, 3] = tvec[0][0]
    pose[1, 3] = tvec[1][0]
    pose[2, 3] = tvec[2][0]

    return pose


def process_data(args):
    global counter
    global output_counter
    global testTrigger

    output_filepath = CFG.REFINE_ITERATIVE_DATA_PATH

    obj = init()

    # parse input
    (id, datalist, isYCB, rot_model) = args
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
                rot_pose = obj.rotateAngle(pose, inplaneRotate)

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

                if CFG.USE_ROUGH_PRED:
                    _, _, _, depth = obj.getLabel()
                    # extract the bounding box
                    bx, by, bw, bh = cv2.boundingRect(target_mask)
                    upperleft = np.array([bx, by])
                    lowerright = np.array([bx + bw, by + bh])

                    high = np.clip(10 / depth, 0, 50)
                    upperleft = (
                        upperleft - np.random.uniform(0, high, upperleft.shape)
                    ).astype(np.int)
                    lowerright = (
                        lowerright + np.random.uniform(0, high, lowerright.shape)
                    ).astype(np.int)
                    random_pose = init_pose_generation(
                        obj, upperleft, lowerright, rot_model, rot_img
                    )
                else:
                    # generate set of random poses
                    random_pose = obj.resamplePose(
                        rot_pose,
                        CFG.OFFSETSAMPLE_VALUE,
                        CFG.DEPTHSAMPLE_VALUE,
                        CFG.ROTATIONSAMPLE_VALUE,
                    )

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

    if CFG.USE_ROUGH_PRED:
        ################### magic net ########################
        viewpt_class = CFG.VIEWPOINT_NUM
        rot_class = CFG.ROTATION_NUM

        rot_model = Magic_Net(viewpt_class=viewpt_class, rot_class=rot_class)
        rot_model.load_state_dict(torch.load(CFG.BEST_MODEL_ROT))
        rot_model.eval()
        rot_model.share_memory()
    else:
        rot_model = None

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

    # image_names = image_names[:10]
    # pose_names = pose_names[:10]

    # generate input for function
    if isYCB:
        datalist = list(zip(image_names, pose_names, mask_names))
    else:
        datalist = list(zip(image_names, pose_names))

    inputP = []
    for o in range(cpu_count()):
        inputP.append((o, list(datalist), isYCB, rot_model))

    with Pool() as p:
        p.imap_unordered(process_data, inputP)
        p.close()
        p.join()

    # process_data(inputP[0])

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
