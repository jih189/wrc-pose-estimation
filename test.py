import numpy as np
import cv2

import src.common.object_model as OM
import src.configuration as CFG

OM.setup(CFG.CAMERA_W, CFG.CAMERA_H)

OM.setProjectMatrixWithIntr(CFG.CAMERA_MATRIX, CFG.CAMERA_W, CFG.CAMERA_H)

obj = OM.ObjectModel()
obj.loadObjectCADModel(CFG.CAD_MODEL)
obj.setIntrinsicMatrix(CFG.CAMERA_MATRIX)

obj.determineSharpEdges(0.5)
obj.generateSamplePoints(0.0001, 0.0001)


def mapt(f, *seq):
    return tuple(map(f, *seq))


test_file = "000100"
frame = cv2.imread(CFG.PROCESSED_DATA_PATH + test_file + ".png")
pose_label = np.load(CFG.PROCESSED_DATA_PATH + test_file + ".npy")


depth = np.linalg.norm(pose_label[:3, 3])


obj.setModelviewMatrix(pose_label)
obj.findVisibleSamplePoint()
viewPoint, inplaneRotation, offsetFromCenter, depth = obj.getLabel()


pre_pose = obj.label2pose(viewPoint, inplaneRotation, offsetFromCenter, depth)
viewPoint = OM.cal_idx(viewPoint)
print("view point = ", viewPoint)
inplaneRotation = inplaneRotation % (2 * np.pi) / (2 * np.pi / 60)
inplaneRotation = int(inplaneRotation)
print("inplane rotation index = ", inplaneRotation)

for pt in obj.sharp_2d_pts:
    pt = mapt(int, pt)
    frame = cv2.circle(frame, pt, radius=0, color=(0, 0, 255), thickness=-1)

originPoint = mapt(int, obj.project3Dto2D((0.0, 0.0, 0.0), pose_label))
xaxis = mapt(int, obj.project3Dto2D((0.05, 0.0, 0.0), pose_label))
frame = cv2.line(frame, originPoint, xaxis, (255, 0, 0), 1)
yaxis = mapt(int, obj.project3Dto2D((0.0, 0.05, 0.0), pose_label))
frame = cv2.line(frame, originPoint, yaxis, (0, 255, 0), 1)
zaxis = mapt(int, obj.project3Dto2D((0.0, 0.0, 0.05), pose_label))
frame = cv2.line(frame, originPoint, zaxis, (0, 0, 255), 1)

obj.setModelviewMatrix(pre_pose)
obj.findVisibleSamplePoint()

for pt in obj.sharp_2d_pts:
    pt = mapt(int, pt)
    frame = cv2.circle(frame, pt, radius=0, color=(0, 255, 0), thickness=-1)

originPoint = mapt(int, obj.project3Dto2D((0.0, 0.0, 0.0), pose_label))
xaxis = mapt(int, obj.project3Dto2D((0.05, 0.0, 0.0), pose_label))
frame = cv2.line(frame, originPoint, xaxis, (255, 0, 0), 1)
yaxis = mapt(int, obj.project3Dto2D((0.0, 0.05, 0.0), pose_label))
frame = cv2.line(frame, originPoint, yaxis, (0, 255, 0), 1)
zaxis = mapt(int, obj.project3Dto2D((0.0, 0.0, 0.05), pose_label))
frame = cv2.line(frame, originPoint, zaxis, (0, 0, 255), 1)

cv2.imshow("view", frame)
cv2.waitKey(0)
