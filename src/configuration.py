# this is the configuration file of pose estimation
import numpy as np
import os

CURRENT_POSE_ESITMATION_DIR = os.getcwd() + "/"

# OBJ_NAME = "pulley"
# IMAGE_SAVE_PATH = CURRENT_POSE_ESITMATION_DIR + "data/images/pulley/"
# VERIFY_IMAGE_PATH = CURRENT_POSE_ESITMATION_DIR + "data/raw/pulley/"
# PROCESSED_DATA_PATH = CURRENT_POSE_ESITMATION_DIR + "data/processed/pulley_rot/"
# REFINE_DATA_PATH = CURRENT_POSE_ESITMATION_DIR + "data/processed/pulley_refine/"
# CAD_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/MBRFA30-2-P6.obj"
# SAMPLE_FACE_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/MBRFA30-2-P6.obj"
# BEST_MODEL_FLOWNET = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_flownet_pulley.pth"
# BEST_MODEL_ROT = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_rot_pulley.pth"
# BEST_MODEL_REFINE = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_refine_pulley.pth"


# OBJ_NAME = "shaft"
# IMAGE_SAVE_PATH = CURRENT_POSE_ESITMATION_DIR + "data/images/shaft/"
# VERIFY_IMAGE_PATH = CURRENT_POSE_ESITMATION_DIR + "data/raw/shaft/"
# PROCESSED_DATA_PATH = CURRENT_POSE_ESITMATION_DIR + "data/processed/shaft_rot/"
# REFINE_DATA_PATH = CURRENT_POSE_ESITMATION_DIR + "data/processed/shaft_refine/"
# CAD_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/SSFHRT10-75-M4-FC55-G20.obj"
# SAMPLE_FACE_MODEL = (
#     CURRENT_POSE_ESITMATION_DIR + "data/mesh/SSFHRT10-75-M4-FC55-G20.obj"
# )
# BEST_MODEL_FLOWNET = (
#     CURRENT_POSE_ESITMATION_DIR + "weights/best_model_flownet_shaft.pth"
# )
# BEST_MODEL_ROT = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_rot_shaft.pth"
# BEST_MODEL_REFINE = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_refine_shaft.pth"
# REFINE_ITERATIVE_DATA_PATH = (
#     CURRENT_POSE_ESITMATION_DIR + "data/processed/shaft_iterative_refine/"
# )
# BEST_MODEL_ITERATIVE_REFINE = (
#     CURRENT_POSE_ESITMATION_DIR + "weights/best_model_iterative_refine_shaft.pth"
# )


# OBJ_NAME = "belt-s-pulley"
# IMAGE_SAVE_PATH = CURRENT_POSE_ESITMATION_DIR + "data/images/belt-s-pulley/"
# VERIFY_IMAGE_PATH = CURRENT_POSE_ESITMATION_DIR + "data/raw/belt-s-pulley/"
# PROCESSED_DATA_PATH = CURRENT_POSE_ESITMATION_DIR + "data/processed/belt-s-pulley_rot/"
# REFINE_DATA_PATH = CURRENT_POSE_ESITMATION_DIR + "data/processed/belt-s-pulley_refine/"
# CAD_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/MBGNA30-2.obj"
# SAMPLE_FACE_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/MBGNA30-2.obj"
# BEST_MODEL_FLOWNET = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_flownet_belt-s-pulley.pth"
# BEST_MODEL_ROT = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_rot_belt-s-pulley.pth"
# BEST_MODEL_REFINE = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_refine_belt-s-pulley.pth"
# REFINE_ITERATIVE_DATA_PATH = CURRENT_POSE_ESITMATION_DIR + "data/processed/belt-s-pulley_iterative_refine/"
# BEST_MODEL_ITERATIVE_REFINE = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_iterative_refine_belt-s-pulley.pth"

# # camera matrix of realsense
# CAMERA_MATRIX = np.array(
#     [
#         [654.968116289191, 0, 322.67377109101744],
#         [0, 657.1436336052552, 248.70937432215163],
#         [0, 0, 1],
#     ],
#     dtype="double",
# )

# OBJ_NAME = "nut"
# IMAGE_SAVE_PATH = CURRENT_POSE_ESITMATION_DIR + "data/images/nut/"
# VERIFY_IMAGE_PATH = CURRENT_POSE_ESITMATION_DIR + "data/raw/nut/"
# PROCESSED_DATA_PATH = CURRENT_POSE_ESITMATION_DIR + "data/processed/nut_rot/"
# REFINE_DATA_PATH = CURRENT_POSE_ESITMATION_DIR + "data/processed/nut_refine/"
# CAD_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/SLBNR6.obj"
# SAMPLE_FACE_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/SLBNR6.obj"
# BEST_MODEL_FLOWNET = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_flownet_nut.pth"
# BEST_MODEL_ROT = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_rot_nut.pth"
# BEST_MODEL_REFINE = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_refine_nut.pth"

OBJ_NAME = "sbar"
IMAGE_SAVE_PATH = CURRENT_POSE_ESITMATION_DIR + "data/images/sbar/"
VERIFY_IMAGE_PATH = CURRENT_POSE_ESITMATION_DIR + "data/raw/sbar/"
PROCESSED_DATA_PATH = CURRENT_POSE_ESITMATION_DIR + "data/processed/sbar_rot/"
REFINE_DATA_PATH = CURRENT_POSE_ESITMATION_DIR + "data/processed/sbar_refine/"
CAD_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/sbar2.obj"
SAMPLE_FACE_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/sbar2.obj"
BEST_MODEL_FLOWNET = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_flownet_sbar.pth"
BEST_MODEL_ROT = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_rot_sbar.pth"
BEST_MODEL_REFINE = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_refine_sbar.pth"
REFINE_ITERATIVE_DATA_PATH = (CURRENT_POSE_ESITMATION_DIR + "data/processed/sbar_iterative_refine/")
#BEST_MODEL_ITERATIVE_REFINE = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_iterative_refine_sbar.pth"
BEST_MODEL_ITERATIVE_REFINE = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_refine_sbar.pth"


#################################### ycb dataset###############################################
# CAMERA_MATRIX = np.array(
#     [[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1],], dtype="double",
# )

# OBJ_NAME = "009_gelatin_box"
# IMAGE_SAVE_PATH = CURRENT_POSE_ESITMATION_DIR + "data/images/009_gelatin_box/"
# VERIFY_IMAGE_PATH = CURRENT_POSE_ESITMATION_DIR + "data/raw/009_gelatin_box/"
# PROCESSED_DATA_PATH = CURRENT_POSE_ESITMATION_DIR + "data/processed/009_gelatin_box_rot/"
# REFINE_DATA_PATH = CURRENT_POSE_ESITMATION_DIR + "data/processed/009_gelatin_box_refine/"
# REFINE_DATA_PATH_CLOSE = CURRENT_POSE_ESITMATION_DIR + "data/processed/009_gelatin_box_refine_close/"
# CAD_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/009_gelatin_box.obj"
# SAMPLE_FACE_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/external/YCB_dataset/models/009_gelatin_box/textured.obj"
# BEST_MODEL_FLOWNET = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_flownet_009_gelatin_box.pth"
# BEST_MODEL_ROT = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_rot_009_gelatin_box.pth"
# BEST_MODEL_REFINE = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_refine_009_gelatin_box.pth"
# BEST_MODEL_REFINE_CLOSE = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_refine_009_gelatin_box_close.pth"

################################ pulley-test##############################################

# OBJ_NAME = "pulley-test"
# IMAGE_SAVE_PATH = CURRENT_POSE_ESITMATION_DIR + "data/images/pulley-test/"
# VERIFY_IMAGE_PATH = CURRENT_POSE_ESITMATION_DIR + "data/raw/pulley-test/"
# PROCESSED_DATA_PATH = CURRENT_POSE_ESITMATION_DIR + "data/processed/pulley-test_rot/"

# REFINE_DATA_PATH = CURRENT_POSE_ESITMATION_DIR + "data/processed/pulley-test_refine/"
# REFINE_ITERATIVE_DATA_PATH = (
#     CURRENT_POSE_ESITMATION_DIR + "data/processed/pulley-test_iterative_refine/"
# )
# CAD_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/MBRFA30-2-P6.obj"
# SAMPLE_FACE_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/MBRFA30-2-P6.obj"
# BEST_MODEL_FLOWNET = (
#     CURRENT_POSE_ESITMATION_DIR + "weights/best_model_flownet_pulley-test.pth"
# )
# BEST_MODEL_ROT = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_rot_pulley-test.pth"
# BEST_MODEL_REFINE = (
#     CURRENT_POSE_ESITMATION_DIR + "weights/best_model_refine_pulley-test.pth"
# )
# BEST_MODEL_ITERATIVE_REFINE = (
#     CURRENT_POSE_ESITMATION_DIR + "weights/best_model_iterative_refine_pulley-test.pth"
# )

# OBJ_NAME = "MBGNA60"
# IMAGE_SAVE_PATH = CURRENT_POSE_ESITMATION_DIR + "data/images/MBGNA60/"
# VERIFY_IMAGE_PATH = CURRENT_POSE_ESITMATION_DIR + "data/raw/MBGNA60/"
# PROCESSED_DATA_PATH = CURRENT_POSE_ESITMATION_DIR + "data/processed/MBGNA60_rot/"
# REFINE_DATA_PATH = CURRENT_POSE_ESITMATION_DIR + "data/processed/MBGNA60_refine/"
# CAD_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/MBGNA60.obj"
# SAMPLE_FACE_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/MBGNA60.obj"
# BEST_MODEL_FLOWNET = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_flownet_MBGNA60.pth"
# BEST_MODEL_ROT = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_rot_MBGNA60.pth"
# BEST_MODEL_REFINE = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_refine_MBGNA60.pth"
# REFINE_ITERATIVE_DATA_PATH = (CURRENT_POSE_ESITMATION_DIR + "data/processed/MBGNA60_iterative_refine/")
# BEST_MODEL_ITERATIVE_REFINE = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_iterative_refine_MBGNA60.pth"

# OBJ_NAME = "MBGNA30"
# IMAGE_SAVE_PATH = CURRENT_POSE_ESITMATION_DIR + "data/images/MBGNA30/"
# VERIFY_IMAGE_PATH = CURRENT_POSE_ESITMATION_DIR + "data/raw/MBGNA30/"
# PROCESSED_DATA_PATH = CURRENT_POSE_ESITMATION_DIR + "data/processed/MBGNA30_rot/"
# REFINE_DATA_PATH = CURRENT_POSE_ESITMATION_DIR + "data/processed/MBGNA30_refine/"
# CAD_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/MBGNA30.obj"
# SAMPLE_FACE_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/MBGNA30.obj"
# BEST_MODEL_FLOWNET = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_flownet_MBGNA30.pth"
# BEST_MODEL_ROT = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_rot_MBGNA30.pth"
# BEST_MODEL_REFINE = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_refine_MBGNA30.pth"
# REFINE_ITERATIVE_DATA_PATH = (CURRENT_POSE_ESITMATION_DIR + "data/processed/MBGNA30_iterative_refine/")
# BEST_MODEL_ITERATIVE_REFINE = CURRENT_POSE_ESITMATION_DIR + "weights/best_model_iterative_refine_MBGNA30.pth"






# camera matrix of wrist camera
# CAMERA_MATRIX = np.array(
#     [
#         [1390.6298269250192, 0, 665.4334864497848],
#         [0, 1389.3521948493603, 314.5310503226418],
#         [0, 0, 1],
#     ],
#     dtype="double",
# )
CAMERA_MATRIX = np.array(
    [
        [1256.98914, 0, 623.943729],
        [0, 1267.18097, 372.156269],
        [0, 0, 1],
    ],
    dtype="double",
)


CAMERA_ID = 0
VIEWPOINT_NUM = 642
# CAMERA_W = 640
# CAMERA_H = 480
CAMERA_W = 1280
CAMERA_H = 720

IMG_SIZE = 240
EXPAND_SIZE = 2.4

LAMBDA_E = 6.0
LAMBDA_V = 6.0

COLOR_AUGMENTATION_BRIGHTNESS = 15
COLOR_AUGMENTATION_CONTRAST = 0.15
