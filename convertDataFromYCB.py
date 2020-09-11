import cv2
from scipy.io import loadmat

# import src.common.object_model as OM
import numpy as np

# importing shutil module
import shutil

from multiprocessing import Pool, Value
import tqdm

MODEL_DIR = "data/external/YCB_dataset/models/"
DATA_DIR = "data/external/YCB_dataset/data/"
IMG_DIR = "data/images/"
OBJ_NAME = "009_gelatin_box"


counter = Value("i", 0)


def loadMeta(mat_file):
    # load meta data
    mat = loadmat(mat_file)
    camera_matrix = mat["intrinsic_matrix"]
    cls_indexes = mat["cls_indexes"]
    poses = mat["poses"]

    return camera_matrix, cls_indexes, poses


def loadClasses(classesfile):
    file1 = open(classesfile, "r")
    Lines = file1.readlines()

    count = 1
    result = []
    # Strips the newline character
    for line in Lines:
        result.append(line.strip())
        count += 1
    return result


def moveData(count, source, destintion, pose, cls_num):

    # copy image
    img_destintion = destintion + "{:06d}".format(count) + ".png"
    shutil.copyfile(source + "-color.png", img_destintion)

    # copy pose
    pose = np.concatenate((pose, [[0, 0, 0, 1]]), 0)
    pose_destintion = destintion + "{:06d}".format(count) + ".npy"
    np.save(pose_destintion, pose)

    # copy mask
    mask_source = source + "-label.png"
    mask_img = cv2.imread(mask_source)

    objmask = np.all(mask_img == [cls_num, cls_num, cls_num], axis=-1)
    othermask = np.all(mask_img != [cls_num, cls_num, cls_num], axis=-1)

    mask_img[objmask] = [255, 255, 255]
    mask_img[othermask] = [0, 0, 0]

    cv2.imwrite(destintion + "{:06d}".format(count) + "-label.png", mask_img)

    # copy bounding box
    box_source = source + "-box.txt"
    dh, dw = mask_img.shape[0], mask_img.shape[1]
    label = []
    with open(box_source) as f:
        line = f.readline()
        while line:
            obj_idx = int(line[:3])
            _, topleftx, toplefty, downrightx, downrighty = line.split()
            topleftx = float(topleftx)
            toplefty = float(toplefty)
            downrightx = float(downrightx)
            downrighty = float(downrighty)
            x = (topleftx + downrightx) / 2.0
            y = (toplefty + downrighty) / 2.0
            w = downrightx - topleftx
            h = downrighty - toplefty
            x = x / dw
            w = w / dw
            y = y / dh
            h = h / dh
            if (
                x >= 0.0
                and y >= 0.0
                and x <= 1.0
                and y <= 1.0
                and w >= 0.0
                and w <= 1.0
                and h >= 0.0
                and h <= 1.0
            ):
                label.append([obj_idx, x, y, w, h])
            line = f.readline()

    with open(destintion + "{:06d}".format(count) + ".txt", "w") as f:
        for data in label:
            line = (
                str(data[0])
                + " "
                + str(data[1])
                + " "
                + str(data[2])
                + " "
                + str(data[3])
                + " "
                + str(data[4])
                + "\n"
            )
            f.write(line)


def processData(args):
    global counter
    filenum, cls_num = args
    metafile = DATA_DIR + filenum + "-meta.mat"
    _, cls_indexes, poses = loadMeta(metafile)
    if np.isin(cls_num, cls_indexes):
        source_file = DATA_DIR + filenum
        dest = IMG_DIR + OBJ_NAME + "/"
        # get pose index
        pose_index = cls_indexes.flatten().tolist().index(cls_num)
        counter_temp = None
        with counter.get_lock():
            counter_temp = counter.value
            counter.value += 1
        moveData(counter_temp, source_file, dest, poses[:, :, pose_index], cls_num)


if __name__ == "__main__":

    classes = loadClasses("data/external/YCB_dataset/image_sets/classes.txt")
    TRAIN_VAL_DIR = "data/external/YCB_dataset/image_sets/trainval.txt"

    cls_num = classes.index(OBJ_NAME) + 1  # begin from 1
    trainvalfile = open(TRAIN_VAL_DIR, "r")

    Lines = trainvalfile.readlines()

    datalist = []
    for l in range(0, len(Lines), 12):
        head_number = int(Lines[l].strip()[:4])
        if head_number < 60:
            datalist.append((Lines[l].strip(), cls_num))

    with Pool() as p:
        for _ in tqdm.tqdm(
            p.imap_unordered(processData, datalist), total=len(datalist)
        ):
            pass
        p.close()
        p.join()
