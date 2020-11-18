import numpy as np
import cv2

import src.common.object_model as OM
import src.configuration as CFG

OM.setup(CFG.CAMERA_W, CFG.CAMERA_H)

OM.setProjectMatrixWithIntr(CFG.CAMERA_MATRIX, CFG.CAMERA_W, CFG.CAMERA_H)

obj = OM.ObjectModel()
obj.loadObjectCADModel(CFG.CAD_MODEL)
obj.setIntrinsicMatrix(CFG.CAMERA_MATRIX)

obj.determineSharpEdges(0.8)
obj.generateSamplePoints(0.00001, 0.00001)


def mapt(f, *seq):
    return tuple(map(f, *seq))


frame = np.zeros((CFG.CAMERA_H, CFG.CAMERA_W, 3))

pose = np.identity(4)
pose[2][3] = 0.5

while True:

    obj.setModelviewMatrix(pose)
    obj.findVisibleSamplePoint()

    for pt in obj.sharp_2d_pts:
        pt = mapt(int, pt)
        frame = cv2.circle(frame, pt, radius=0, color=(255, 255, 255), thickness=-1)

    cv2.imshow("view", frame)
    cv2.waitKey(0)
