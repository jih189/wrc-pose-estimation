# this is the configuration file of pose estimation
import numpy as np

CAMERA_MATRIX = np.array(
    [
        [654.968116289191, 0, 322.67377109101744],
        [0, 657.1436336052552, 248.70937432215163],
        [0, 0, 1],
    ],
    dtype="double",
)

IMAGE_SAVE_PATH = "data/images/pulley/"

CAD_MODEL = "data/mesh/MBRFA30-2-P6.obj"

CAMERA_ID = 4
CAMERA_W = 640
CAMERA_H = 480
