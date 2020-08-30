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
from multiprocessing import Pool, Value, cpu_count, Array

counter = Value("i", 0)
output_counter = Value("i", 0)
vparr = Array("i", [0] * 64)
rotarr = Array("i", [0] * 60)


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


def process_data(args):
    global counter
    global output_counter
    global vparr
    global rotarr

    output_filepath = CFG.PROCESSED_DATA_PATH

    obj = init()

    # parse input
    (id, datalist) = args
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

        # read image and pose
        img = cv2.imread(img_names[current_index])
        img = np.array(img)

        pose = np.load(pose_names[current_index])
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

        if (
            int(upperleft_rand[1]) < 0
            or int(lowerright_rand[1]) >= img.shape[0]
            or int(upperleft_rand[0]) < 0
            or int(lowerright_rand[0]) > img.shape[1]
        ):
            continue

        viewPoint, inplaneRotation, offsetFromCenter, depth = obj.getLabel()
        inplaneRotation = inplaneRotation % (2 * np.pi) / (2 * np.pi / 60)
        if np.isnan(inplaneRotation):
            continue

        with output_counter.get_lock():
            current_output_index = output_counter.value
            output_counter.value += 1

        cropImg = img[
            int(upperleft_rand[1]) : int(lowerright_rand[1]),
            int(upperleft_rand[0]) : int(lowerright_rand[0]),
        ]

        np.save(
            output_filepath
            + "bounding"
            + "{:06d}".format(current_output_index)
            + ".npy",
            np.array(
                [
                    int(upperleft_rand[0]),
                    int(upperleft_rand[1]),
                    int(lowerright_rand[0]),
                    int(lowerright_rand[1]),
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

    # image_names = image_names[:10]
    # pose_names = pose_names[:10]

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
