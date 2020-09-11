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
from multiprocessing import Pool, Value, cpu_count, Array

EXPAND_SIZE = 2.0
RANDOM_NUM = 4

counter = Value("i", 0)
output_counter = Value("i", 0)


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


def process_data(args):
    global counter
    global output_counter

    output_filepath = CFG.REFINE_DATA_PATH

    obj = init()

    # parse input
    (id, datalist, isYCB) = args
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

        # update the progress bar
        progress = int(50.0 * current_index / len(datalist))
        rest_progress = 50 - progress
        print(
            "Progress: ["
            + "=" * progress
            + " " * rest_progress
            + "]"
            + str(100.0 * current_index / len(datalist))
            + "%"
            + " index "
            + str(current_index),
            end="\r",
            flush=True,
        )

        # read image and pose
        img = cv2.imread(img_names[current_index])
        pose = np.load(pose_names[current_index])

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
            target_mask = cv2.imread(mask_names[current_index])
        else:
            # mask bit
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

            if ex < 0 or ey < 0 or ex + ew >= img.shape[1] or ey + eh >= img.shape[0]:
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
            if init3dpts.shape[0] < 500:
                print("no enough sample point in this pose! continue...")
                continue

            # get the 3d sample pts from target pose
            obj.setModelviewMatrix(target_pose_at_center)
            # generate edge of on the object
            obj.findVisibleSamplePoint()

            target3dpts = np.array(obj.visible_sharpedge_samplepoint)
            if target3dpts.shape[0] < 500:
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
                output_filepath + "{:06d}".format(current_output_index) + "flow.png",
                crop_flowImg,
            )

            cv2.imwrite(
                output_filepath + "{:06d}".format(current_output_index) + "3d.png",
                crop_3dImg,
            )

            cv2.imwrite(
                output_filepath + "{:06d}".format(current_output_index) + "mask.png",
                crop_mask,
            )
            cv2.imwrite(
                output_filepath + "{:06d}".format(current_output_index) + "edge.png",
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

    # image_names = image_names[886:]
    # pose_names = pose_names[886:]
    # mask_names = mask_names[886:]

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
