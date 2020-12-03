# this is the configuration file of pose estimation
import numpy as np
import os

CURRENT_POSE_ESITMATION_DIR = os.getcwd() + "/"

# OBJ_NAME = "pulley"
# CAD_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/MBRFA30-2-P6.obj"
# SAMPLE_FACE_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/MBRFA30-2-P6.obj"


OBJ_NAME = "shaft"
CAD_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/SSFHRT10-75-M4-FC55-G20.obj"
SAMPLE_FACE_MODEL = (
    CURRENT_POSE_ESITMATION_DIR + "data/mesh/SSFHRT10-75-M4-FC55-G20.obj"
)

# OBJ_NAME = "belt-s-pulley"
# CAD_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/MBGNA30-2.obj"
# SAMPLE_FACE_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/MBGNA30-2.obj"

# OBJ_NAME = "nut"
# CAD_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/SLBNR6.obj"
# SAMPLE_FACE_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/SLBNR6.obj"

# OBJ_NAME = "housing"
# CAD_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/SBARB6200ZZ-30.obj"
# SAMPLE_FACE_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/SBARB6200ZZ-30.obj"

# OBJ_NAME = "pulley-test"
# CAD_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/MBRFA30-2-P6.obj"
# SAMPLE_FACE_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/MBRFA30-2-P6.obj"

IMAGE_SAVE_PATH = CURRENT_POSE_ESITMATION_DIR + "data/images/" + OBJ_NAME + "/"
VERIFY_IMAGE_PATH = CURRENT_POSE_ESITMATION_DIR + "data/raw/" + OBJ_NAME + "/"
PROCESSED_DATA_PATH = (
    CURRENT_POSE_ESITMATION_DIR + "data/processed/" + OBJ_NAME + "_rot/"
)
REFINE_DATA_PATH = (
    CURRENT_POSE_ESITMATION_DIR + "data/processed/" + OBJ_NAME + "_refine/"
)
BEST_MODEL_FLOWNET = (
    CURRENT_POSE_ESITMATION_DIR + "weights/best_model_flownet_" + OBJ_NAME + ".pth"
)
BEST_MODEL_ROT = (
    CURRENT_POSE_ESITMATION_DIR + "weights/best_model_rot_" + OBJ_NAME + ".pth"
)
BEST_MODEL_REFINE = (
    CURRENT_POSE_ESITMATION_DIR + "weights/best_model_refine_" + OBJ_NAME + ".pth"
)
REFINE_ITERATIVE_DATA_PATH = (
    CURRENT_POSE_ESITMATION_DIR + "data/processed/" + OBJ_NAME + "_iterative_refine/"
)
BEST_MODEL_ITERATIVE_REFINE = (
    CURRENT_POSE_ESITMATION_DIR
    + "weights/best_model_iterative_refine_"
    + OBJ_NAME
    + ".pth"
)

######################################## camera information ######################################################

# # camera matrix of realsense
# CAMERA_MATRIX = np.array(
#     [
#         [654.968116289191, 0, 322.67377109101744],
#         [0, 657.1436336052552, 248.70937432215163],
#         [0, 0, 1],
#     ],
#     dtype="double",
# )


# camera matrix of wrist camera
CAMERA_MATRIX = np.array(
    [
        [1390.6298269250192, 0, 665.4334864497848],
        [0, 1389.3521948493603, 314.5310503226418],
        [0, 0, 1],
    ],
    dtype="double",
)

CAMERA_ID = 0
# CAMERA_W = 640
# CAMERA_H = 480
CAMERA_W = 1280
CAMERA_H = 720

################################## hyper-parameters #######################################################
VIEWPOINT_NUM = 642
ROTATION_NUM = 60
IMG_SIZE = 240
EXPAND_SIZE = 2.4

LAMBDA_E = 1.0
LAMBDA_V = 1.0

COLOR_AUGMENTATION_BRIGHTNESS = 15
COLOR_AUGMENTATION_CONTRAST = 0.15

USE_ROUGH_PRED = False  # do not set this to true yet

OFFSETSAMPLE_VALUE = 0.015
DEPTHSAMPLE_VALUE = 0.03
ROTATIONSAMPLE_VALUE = 0.15
