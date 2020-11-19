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
obj.generateSamplePoints(0.00001)


def mapt(f, *seq):
    return tuple(map(f, *seq))


blackImg = np.zeros((CFG.CAMERA_H, CFG.CAMERA_W, 3))

pose = np.identity(4)
pose[2][3] = 0.5

transform = np.identity(4)
transform[:3, :3] = (R.from_euler("z", 0.1)).as_matrix()

for i in range(10):
    frame = blackImg.copy()

    pose = np.dot(transform, pose)

    render_s = time.time()
    obj.setModelviewMatrix(pose)

    # obj.renderVisibleFaces()
    # mask = obj.getMask()
    # frame[mask] = 255

    obj.findVisibleSamplePoint()
    # obj.render()
    # OM.testInPygame()
    for pt in obj.sharp_2d_pts:
        pt = mapt(int, pt)
        frame = cv2.circle(frame, pt, radius=0, color=(255, 255, 255), thickness=-1)
    render_e = time.time()
    print("render time = ", render_e - render_s)

    cv2.imshow("view", frame)
    ch = cv2.waitKey(0)
    if ch & 0xFF == ord("q"):
        break
