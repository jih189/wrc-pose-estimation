# this is the configuration file of pose estimation
import numpy as np


# OBJ_NAME = "pulley"
# IMAGE_SAVE_PATH = "data/images/pulley/"
# VERIFY_IMAGE_PATH = "data/raw/pulley/"
# PROCESSED_DATA_PATH = "data/processed/pulley_rot/"
# REFINE_DATA_PATH = "data/processed/pulley_refine/"
# CAD_MODEL = "data/mesh/MBRFA30-2-P6.obj"
# SAMPLE_FACE_MODEL = "data/mesh/MBRFA30-2-P6.obj"
# BEST_MODEL_FLOWNET = "weights/best_model_flownet_pulley.pth"
# BEST_MODEL_ROT = "weights/best_model_rot_pulley.pth"
# BEST_MODEL_REFINE = "weights/best_model_refine_pulley.pth"


OBJ_NAME = "shaft"
IMAGE_SAVE_PATH = "data/images/shaft/"
VERIFY_IMAGE_PATH = "data/raw/shaft/"
PROCESSED_DATA_PATH = "data/processed/shaft_rot/"
REFINE_DATA_PATH = "data/processed/shaft_refine/"
CAD_MODEL = "data/mesh/SSFHRT10-75-M4-FC55-G20.obj"
SAMPLE_FACE_MODEL = "data/mesh/SSFHRT10-75-M4-FC55-G20.obj"
BEST_MODEL_FLOWNET = "weights/best_model_flownet_shaft.pth"
BEST_MODEL_ROT = "weights/best_model_rot_shaft.pth"
BEST_MODEL_REFINE = "weights/best_model_refine_shaft.pth"
REFINE_ITERATIVE_DATA_PATH = (
    "/home/cogrob-wrc/wrc-pose-estimation/data/processed/shaft_iterative_refine/"
)
BEST_MODEL_ITERATIVE_REFINE = (
    "/home/cogrob-wrc/wrc-pose-estimation/weights/best_model_iterative_refine_shaft.pth"
)


# OBJ_NAME = "belt-s-pulley"
# IMAGE_SAVE_PATH = "data/images/belt-s-pulley/"
# VERIFY_IMAGE_PATH = "data/raw/belt-s-pulley/"
# PROCESSED_DATA_PATH = "data/processed/belt-s-pulley_rot/"
# REFINE_DATA_PATH = "data/processed/belt-s-pulley_refine/"
# CAD_MODEL = "data/mesh/MBGNA30-2.obj"
# SAMPLE_FACE_MODEL = "data/mesh/MBGNA30-2.obj"
# BEST_MODEL_FLOWNET = "weights/best_model_flownet_belt-s-pulley.pth"
# BEST_MODEL_ROT = "weights/best_model_rot_belt-s-pulley.pth"
# BEST_MODEL_REFINE = "weights/best_model_refine_belt-s-pulley.pth"

# # camera matrix of realsense
CAMERA_MATRIX = np.array(
    [
        [654.968116289191, 0, 322.67377109101744],
        [0, 657.1436336052552, 248.70937432215163],
        [0, 0, 1],
    ],
    dtype="double",
)

# OBJ_NAME = "nut"
# IMAGE_SAVE_PATH = "data/images/nut/"
# VERIFY_IMAGE_PATH = "data/raw/nut/"
# PROCESSED_DATA_PATH = "data/processed/nut_rot/"
# REFINE_DATA_PATH = "data/processed/nut_refine/"
# CAD_MODEL = "data/mesh/SLBNR6.obj"
# SAMPLE_FACE_MODEL = "data/mesh/SLBNR6.obj"
# BEST_MODEL_FLOWNET = "weights/best_model_flownet_nut.pth"
# BEST_MODEL_ROT = "weights/best_model_rot_nut.pth"
# BEST_MODEL_REFINE = "weights/best_model_refine_nut.pth"

# OBJ_NAME = "housing"
# IMAGE_SAVE_PATH = "data/images/housing/"
# VERIFY_IMAGE_PATH = "data/raw/housing/"
# PROCESSED_DATA_PATH = "data/processed/housing_rot/"
# REFINE_DATA_PATH = "data/processed/housing_refine/"
# CAD_MODEL = "data/mesh/SBARB6200ZZ-30.obj"
# SAMPLE_FACE_MODEL = "data/mesh/SBARB6200ZZ-30.obj"
# BEST_MODEL_FLOWNET = "weights/best_model_flownet_housing.pth"
# BEST_MODEL_ROT = "weights/best_model_rot_housing.pth"
# BEST_MODEL_REFINE = "weights/best_model_refine_housing.pth"


#################################### ycb dataset###############################################
# CAMERA_MATRIX = np.array(
#     [[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1],], dtype="double",
# )

# OBJ_NAME = "009_gelatin_box"
# IMAGE_SAVE_PATH = "data/images/009_gelatin_box/"
# VERIFY_IMAGE_PATH = "data/raw/009_gelatin_box/"
# PROCESSED_DATA_PATH = "data/processed/009_gelatin_box_rot/"
# REFINE_DATA_PATH = "data/processed/009_gelatin_box_refine/"
# REFINE_DATA_PATH_CLOSE = "data/processed/009_gelatin_box_refine_close/"
# CAD_MODEL = "data/mesh/009_gelatin_box.obj"
# SAMPLE_FACE_MODEL = "data/external/YCB_dataset/models/009_gelatin_box/textured.obj"
# BEST_MODEL_FLOWNET = "weights/best_model_flownet_009_gelatin_box.pth"
# BEST_MODEL_ROT = "weights/best_model_rot_009_gelatin_box.pth"
# BEST_MODEL_REFINE = "weights/best_model_refine_009_gelatin_box.pth"
# BEST_MODEL_REFINE_CLOSE = "weights/best_model_refine_009_gelatin_box_close.pth"

################################ pulley-test##############################################

# OBJ_NAME = "pulley-test"
# IMAGE_SAVE_PATH = "/home/cogrob-wrc/wrc-pose-estimation/data/images/pulley-test/"
# VERIFY_IMAGE_PATH = "/home/cogrob-wrc/wrc-pose-estimation/data/raw/pulley-test/"
# PROCESSED_DATA_PATH = (
#     "/home/cogrob-wrc/wrc-pose-estimation/data/processed/pulley-test_rot/"
# )
# REFINE_DATA_PATH = (
#     "/home/cogrob-wrc/wrc-pose-estimation/data/processed/pulley-test_refine/"
# )
# REFINE_ITERATIVE_DATA_PATH = (
#     "/home/cogrob-wrc/wrc-pose-estimation/data/processed/pulley-test_iterative_refine/"
# )
# CAD_MODEL = "/home/cogrob-wrc/wrc-pose-estimation/data/mesh/MBRFA30-2-P6.obj"
# SAMPLE_FACE_MODEL = "/home/cogrob-wrc/wrc-pose-estimation/data/mesh/MBRFA30-2-P6.obj"
# BEST_MODEL_FLOWNET = (
#     "/home/cogrob-wrc/wrc-pose-estimation/weights/best_model_flownet_pulley-test.pth"
# )
# BEST_MODEL_ROT = (
#     "/home/cogrob-wrc/wrc-pose-estimation/weights/best_model_rot_pulley-test.pth"
# )
# BEST_MODEL_REFINE = (
#     "/home/cogrob-wrc/wrc-pose-estimation/weights/best_model_refine_pulley-test.pth"
# )
# BEST_MODEL_ITERATIVE_REFINE = "/home/cogrob-wrc/wrc-pose-estimation/weights/best_model_iterative_refine_pulley-test.pth"


# # camera matrix of wrist camera
# CAMERA_MATRIX = np.array(
#     [
#         [1390.6298269250192, 0, 665.4334864497848],
#         [0, 1389.3521948493603, 314.5310503226418],
#         [0, 0, 1],
#     ],
#     dtype="double",
# )

CAMERA_ID = 4
VIEWPOINT_NUM = 642
CAMERA_W = 640
CAMERA_H = 480
# CAMERA_W = 1280
# CAMERA_H = 720

IMG_SIZE = 240
EXPAND_SIZE = 2.4
