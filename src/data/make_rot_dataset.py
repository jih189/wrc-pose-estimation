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


@click.command()
@click.argument(
    "input_filepath", default="data/raw/pulley/", type=click.Path(exists=True)
)
@click.argument(
    "output_filepath", default="data/processed/pulley_rot/", type=click.Path()
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
        img = np.array(img)

        pose = np.load(pose_name)
        depth = pose[2, 3]

        # generate the real bounding box for object
        obj.setModelviewMatrix(pose)

        upperleft, lowerright = obj.findVisibleSamplePoint()
        upperleft, lowerright = (
            np.array(upperleft).reshape(2),
            np.array(lowerright).reshape(2),
        )

        upperleft_nonrand, lowerright_nonrand = get_centered_crop(upperleft, lowerright)

        high = np.clip(10 / depth, 0, 50)
        crop_upperleft = (
            upperleft - np.random.uniform(0, high, upperleft.shape)
        ).astype(np.int)
        crop_lowerright = (
            lowerright + np.random.uniform(0, high, lowerright.shape)
        ).astype(np.int)

        upperleft_rand, lowerright_rand = get_centered_crop(
            crop_upperleft, crop_lowerright
        )

        try:
            cropImg = img[
                int(upperleft_rand[1]) : int(lowerright_rand[1]),
                int(upperleft_rand[0]) : int(lowerright_rand[0]),
            ]
            np.save(
                output_filepath + "bounding" + "{:06d}".format(current_index) + ".npy",
                np.array(
                    [
                        int(upperleft_rand[0]),
                        int(upperleft_rand[1]),
                        int(lowerright_rand[0]),
                        int(lowerright_rand[1]),
                    ]
                ),
            )
        except IndexError:
            cropImg = img[
                int(upperleft_nonrand[1]) : int(lowerright_nonrand[1]),
                int(upperleft_nonrand[0]) : int(lowerright_nonrand[0]),
            ]
            np.save(
                output_filepath + "bounding" + "{:06d}".format(current_index) + ".npy",
                np.array(
                    [
                        int(upperleft_nonrand[0]),
                        int(upperleft_nonrand[1]),
                        int(lowerright_nonrand[0]),
                        int(lowerright_nonrand[1]),
                    ]
                ),
            )

        cv2.imwrite(
            output_filepath + "crop" + "{:06d}".format(current_index) + ".png", cropImg
        )
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
