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

# CAMERA_MATRIX = np.array(
#     [[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1],], dtype="double",
# )

# IMAGE_SAVE_PATH = "data/images/pulley/"
# VERIFY_IMAGE_PATH = "data/raw/pulley/"
# PROCESSED_DATA_PATH = "data/processed/pulley_rot/"
# REFINE_SATA_PATH = "data/processed/pulley_refine/"
# CAD_MODEL = "data/mesh/MBRFA30-2-P6.obj"

# IMAGE_SAVE_PATH = "data/images/housing/"
# VERIFY_IMAGE_PATH = "data/raw/housing/"
# PROCESSED_DATA_PATH = "data/processed/housing_rot/"
# REFINE_SATA_PATH = "data/processed/housing_refine/"
# CAD_MODEL = "data/mesh/SBARB6200ZZ-30.obj"

IMAGE_SAVE_PATH = "data/images/009_gelatin_box/"
VERIFY_IMAGE_PATH = "data/raw/009_gelatin_box/"
PROCESSED_DATA_PATH = "data/processed/009_gelatin_box_rot/"
REFINE_SATA_PATH = "data/processed/009_gelatin_box_refine/"
CAD_MODEL = "data/mesh/009_gelatin_box.obj"
ROT_WEIGHTS = [
    0.01282051282051282,
    1.0,
    1.0,
    0.00510204081632653,
    1.0,
    0.05263157894736842,
    0.018518518518518517,
    1.0,
    0.00423728813559322,
    0.02564102564102564,
    1.0,
    0.041666666666666664,
    1.0,
    0.007407407407407408,
    0.007352941176470588,
    1.0,
    0.005952380952380952,
    0.06666666666666667,
    1.0,
    0.05555555555555555,
    1.0,
    0.02702702702702703,
    0.007692307692307693,
    1.0,
    1.0,
    1.0,
    0.024390243902439025,
    0.006944444444444444,
    1.0,
    0.005208333333333333,
    0.05555555555555555,
    1.0,
    1.0,
    1.0,
    1.0,
    0.007194244604316547,
    1.0,
    0.029411764705882353,
    1.0,
    1.0,
    0.011111111111111112,
    1.0,
    0.018518518518518517,
    0.14285714285714285,
    1.0,
    0.03333333333333333,
    1.0,
    1.0,
    0.005847953216374269,
    0.0625,
    0.09090909090909091,
    1.0,
    1.0,
    0.06666666666666667,
    0.05,
    1.0,
    0.012658227848101266,
    1.0,
    0.022727272727272728,
    0.125,
    1.0,
    0.029411764705882353,
    0.07142857142857142,
    1.0,
]

CAMERA_ID = 4
CAMERA_W = 640
CAMERA_H = 480
