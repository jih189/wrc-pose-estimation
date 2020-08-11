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

# IMAGE_SAVE_PATH = "data/images/pulley/"
# VERIFY_IMAGE_PATH = "data/raw/pulley_verify/"
# CAD_MODEL = "data/mesh/MBRFA30-2-P6.obj"

IMAGE_SAVE_PATH = "data/images/housing/"
VERIFY_IMAGE_PATH = "data/raw/housing/"
PROCESSED_DATA_PATH = "data/processed/housing_rot/"
REFINE_SATA_PATH = "data/processed/housing_refine/"
CAD_MODEL = "data/mesh/SBARB6200ZZ-30.obj"

CAMERA_ID = 4
CAMERA_W = 640
CAMERA_H = 480
