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

EXPAND_SIZE = 2.0
RANDOM_NUM = 2


def init():
    # load the object mesh
    OM.setup(CFG.CAMERA_W, CFG.CAMERA_H)
    OM.setProjectMatrixWithIntr(CFG.CAMERA_MATRIX, CFG.CAMERA_W, CFG.CAMERA_H)

    obj = OM.ObjectModel()
    obj.setIntrinsicMatrix(CFG.CAMERA_MATRIX)
    obj.loadObjectCADModel(CFG.CAD_MODEL)

    obj.determineSharpEdges(0.05)
    obj.generateSamplePoints(0.0001, 0.001)
    return obj


@click.command()
@click.argument(
    "input_filepath", default="data/raw/pulley/", type=click.Path(exists=True)
)
@click.argument(
    "output_filepath", default="data/processed/pulley_refine/", type=click.Path()
)
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    logger.info(f"Input directory: {input_filepath}")
    logger.info(f"Output directory: {output_filepath}")

    input_path = Path(input_filepath)
    image_names, pose_names = [], []
    for f in input_path.iterdir():
        if f.match("*.png"):
            image_names.append(str(f))
        if f.match("*.npy"):
            pose_names.append(str(f))
    image_names.sort()
    pose_names.sort()

    obj = init()

    current_index = 0
    for img_name, pose_name in tqdm(zip(image_names, pose_names)):
        # read image and pose
        img = cv2.imread(img_name)
        pose = np.load(pose_name)

        # generate set of random poses
        random_poses = obj.resample(pose, RANDOM_NUM)

        # generate the real bounding box for object
        obj.setModelviewMatrix(pose)
        obj.findVisibleSamplePoint()

        # mask bit
        target_mask = obj.getVisibleArea(img)

        # find the crop size
        # [rx, ry, rw, rh] = cv2.boundingRect(target_mask)

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
            mask = obj.getVisibleArea(img)

            # get edge img
            edge = obj.getEdge(img.shape[0], img.shape[1])

            # find the crop size
            [x, y, w, h] = cv2.boundingRect(mask)

            boundingsize = max(w, h) * EXPAND_SIZE

            ex = int(x + (w - boundingsize) / 2)
            ey = int(y + (h - boundingsize) / 2)
            ew = int(boundingsize)
            eh = int(boundingsize)

            if ew == 0 or eh == 0:
                logger.warn(f"Invalid image width/height. Continuing...")
                continue

            if ex < 0 or ey < 0 or ex + ew >= img.shape[1] or ey + eh >= img.shape[0]:
                logger.warn(f"Bounding box out of image. Continuing...")
                continue

            # crop_img = img[ey : ey + eh, ex : ex + ew].copy()
            # cropped image with initial pose as center
            crop_img = img[ey : ey + eh, ex : ex + ew]
            # cropped mask for initial pose
            crop_mask = mask[ey : ey + eh, ex : ex + ew]
            # cropped edges for initial pose
            crop_edge = edge[ey : ey + eh, ex : ex + ew]

            # bounding box for target pose
            # crop_bounding = np.zeros((eh, ew), np.uint8)
            # crop_bounding[
            #     max(0, ry - ey) : min(eh, ry - ey + rh),
            #     max(0, rx - ex) : min(ew, rx - ex + rw),
            # ] = 255

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
            if init3dpts.shape[0] < 1000:
                continue

            # get the 3d sample pts from target pose
            obj.setModelviewMatrix(target_pose_at_center)
            # generate edge of on the object
            obj.findVisibleSamplePoint()

            target3dpts = np.array(obj.visible_sharpedge_samplepoint)
            if target3dpts.shape[0] < 1000:
                continue

            cv2.imwrite(
                output_filepath + "{:06d}".format(current_index) + "img.png", crop_img,
            )
            cv2.imwrite(
                output_filepath + "{:06d}".format(current_index) + "mask.png",
                crop_mask,
            )
            cv2.imwrite(
                output_filepath + "{:06d}".format(current_index) + "edge.png",
                crop_edge,
            )
            cv2.imwrite(
                output_filepath + "{:06d}".format(current_index) + "labelmask.png",
                crop_label_mask,
            )

            np.save(
                output_filepath + "{:06d}".format(current_index) + "initPose.npy",
                current_pose_at_center,
            )
            np.save(
                output_filepath + "{:06d}".format(current_index) + "targetPose.npy",
                target_pose_at_center,
            )
            np.save(
                output_filepath + "{:06d}".format(current_index) + "init3dPt.npy",
                init3dpts,
            )
            np.save(
                output_filepath + "{:06d}".format(current_index) + "target3dPt.npy",
                target3dpts,
            )

            if current_index % 500 == 0:
                logger.info(current_index)

            current_index += 1
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
