# this is the configuration file of pose estimation
import numpy as np

# CAMERA_MATRIX = np.array(
#     [
#         [654.968116289191, 0, 322.67377109101744],
#         [0, 657.1436336052552, 248.70937432215163],
#         [0, 0, 1],
#     ],
#     dtype="double",
# )

CAMERA_MATRIX = np.array(
    [[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1],], dtype="double",
)

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
# BEST_MODEL_FLOWNET = "weights/best_model_flownet_housing.pth"
# BEST_MODEL_ROT = "weights/best_model_rot_housing.pth"
# BEST_MODEL_FLOWNET = "weights/best_model_refine_housing.pth"

IMAGE_SAVE_PATH = "data/images/009_gelatin_box/"
VERIFY_IMAGE_PATH = "data/raw/009_gelatin_box/"
PROCESSED_DATA_PATH = "data/processed/009_gelatin_box_rot/"
REFINE_SATA_PATH = "data/processed/009_gelatin_box_refine/"
CAD_MODEL = "data/mesh/009_gelatin_box.obj"
BEST_MODEL_FLOWNET = "weights/best_model_flownet_009_gelatin_box.pth"
BEST_MODEL_ROT = "weights/best_model_rot_009_gelatin_box.pth"
BEST_MODEL_REFINE = "weights/best_model_refine_009_gelatin_box.pth"

CAMERA_ID = 4
CAMERA_W = 640
CAMERA_H = 480
