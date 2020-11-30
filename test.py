import numpy as np
import cv2
import time

import src.common.object_model as OM
import src.configuration as CFG
from scipy.spatial.transform import Rotation as R

OM.setup(CFG.CAMERA_W, CFG.CAMERA_H)

OM.setProjectMatrixWithIntr(CFG.CAMERA_MATRIX, CFG.CAMERA_W, CFG.CAMERA_H)

obj = OM.ObjectModel()
obj.loadObjectCADModel(CFG.CAD_MODEL)
obj.setIntrinsicMatrix(CFG.CAMERA_MATRIX)

obj.determineSharpEdges(0.8)
obj.generateSamplePoints(0.001)

objectid = "000000"

img = cv2.imread(CFG.PROCESSED_DATA_PATH + objectid + ".png")
pose = np.load(CFG.PROCESSED_DATA_PATH + objectid + ".npy")
center_pt = obj.project3Dto2D((0, 0, 0), pose)
center_pt = (int(center_pt[0]), int(center_pt[1]))

obj.setModelviewMatrix(pose)
obj.findVisibleSamplePoint()

for c in obj.sharp_2d_pts:
    img = cv2.circle(
        img,
        (int(c[0]), int(c[1])),
        radius=1,
        color=(0, 0, 255),
        thickness=-1,
    )

for r in range(8):
    inplaneRotate = r * 45.0
    rot_img = OM.rotate_image(img, inplaneRotate, center_pt)
    rot_pose = obj.rotateAngle(pose, inplaneRotate)

    cornerpoints_2d = obj.getCornerPoints(pose)
    objectPoint = obj.project3Dto2D((0,0,0), pose)
    rp = []
    for p in cornerpoints_2d:
        rp.append(obj.rotate_f(objectPoint, p, -np.radians(inplaneRotate)))

    for p in rp:
        rot_img = cv2.circle(
            rot_img,
            (int(p[0]), int(p[1])),
            radius=3,
            color=(255, 255, 0),
            thickness=-1,
        )

    # generate the real bounding box for object
    obj.setModelviewMatrix(rot_pose)
    obj.findVisibleSamplePoint()

    for c in obj.sharp_2d_pts:
        rot_img = cv2.circle(
            rot_img,
            (int(c[0]), int(c[1])),
            radius=1,
            color=(0, 255, 0),
            thickness=-1,
        )
    cv2.imshow("window", rot_img)
    cv2.waitKey(0)
