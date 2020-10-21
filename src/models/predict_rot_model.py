import numpy as np
import cv2
import torch
import torch.nn as nn

from torch.autograd import Variable

from models.model import Magic_Net
import src.common.object_model as OM
import src.configuration as CFG


def mapt(f, *seq):
    return tuple(map(f, *seq))


OM.setup(CFG.CAMERA_W, CFG.CAMERA_H)

OM.setProjectMatrixWithIntr(CFG.CAMERA_MATRIX, CFG.CAMERA_W, CFG.CAMERA_H)

obj = OM.ObjectModel()
obj.loadObjectCADModel(CFG.CAD_MODEL)
obj.setIntrinsicMatrix(CFG.CAMERA_MATRIX)

obj.determineSharpEdges(0.8)
obj.generateSamplePoints(0.0001, 0.0001)

# load data
test_file = "000001"
frame = cv2.imread(CFG.PROCESSED_DATA_PATH + test_file + ".png")
img_path = CFG.PROCESSED_DATA_PATH + "crop" + test_file + ".png"
boundingbox = np.load(CFG.PROCESSED_DATA_PATH + "bounding" + test_file + ".npy")
img = cv2.imread(img_path)
img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_AREA)
img = img[:, :, :3].transpose(2, 0, 1)
img = img[np.newaxis, ...]

input = Variable(torch.from_numpy(img).cuda()).float()

viewpt_class = 64
rot_class = 60

model = Magic_Net(viewpt_class=viewpt_class, rot_class=rot_class).cuda()
model = nn.DataParallel(model)
model = torch.load(CFG.BEST_MODEL_ROT)
model.eval()

# predict offset, view point, and inplane rotation
output = model(input)

pred = output[:, :viewpt_class].data.cpu().numpy()
pred = np.argmax(pred, axis=1)

# load target information
pose_label = np.load(CFG.PROCESSED_DATA_PATH + test_file + ".npy")
depth = np.linalg.norm(pose_label[:3, 3])

obj.setModelviewMatrix(pose_label)
obj.findVisibleSamplePoint()
viewPoint, inplaneRotation, offsetFromCenter, depth = obj.getLabel()

print("label viewpoint index ", OM.cal_idx(viewPoint))
print(
    "label inplanerotation index ", int(inplaneRotation % (2 * np.pi) / (np.pi / 30)),
)

# convert it to index then back to number
# todo

label_img = frame.copy()
updated_label_pose = obj.label2pose(
    viewPoint, -inplaneRotation, offsetFromCenter, depth
)

obj.setModelviewMatrix(updated_label_pose)
obj.findVisibleSamplePoint()
for pt in obj.sharp_2d_pts:
    pt = mapt(int, pt)
    label_img = cv2.circle(label_img, pt, radius=1, color=(0, 255, 0), thickness=-1)

originPoint = mapt(int, obj.project3Dto2D((0.0, 0.0, 0.0), updated_label_pose))
xaxis = mapt(int, obj.project3Dto2D((0.05, 0.0, 0.0), updated_label_pose))
label_img = cv2.line(label_img, originPoint, xaxis, (255, 0, 0), 1)
yaxis = mapt(int, obj.project3Dto2D((0.0, 0.05, 0.0), updated_label_pose))
label_img = cv2.line(label_img, originPoint, yaxis, (0, 255, 0), 1)
zaxis = mapt(int, obj.project3Dto2D((0.0, 0.0, 0.05), updated_label_pose))
label_img = cv2.line(label_img, originPoint, zaxis, (0, 0, 255), 1)

cv2.imshow("label", label_img)
cv2.waitKey(0)

# # extract offset, view point, and inplane rotation from pred
# upperleftx, upperlefty, lowerrightx, lowerrighty = (
#     boundingbox[0].astype(np.int),
#     boundingbox[1].astype(np.int),
#     boundingbox[2].astype(np.int),
#     boundingbox[3].astype(np.int),
# )
# l = lowerrightx - upperleftx
# principle_pt = np.array([CFG.CAMERA_MATRIX[0, 2], CFG.CAMERA_MATRIX[1, 2]])

# position = torch.sigmoid(output[:, viewpt_class + rot_class :]).data.cpu().numpy()
# position *= l
# offset = position[:, :2]

# offset = np.array([upperleftx, upperlefty]) + offset.reshape(2) - principle_pt

# viewpt = np.array(OM.idx2vp(pred[0]))
# print("pred vp index ", pred[0])

# rot = output[:, viewpt_class : viewpt_class + rot_class].data.cpu().numpy()
# rot = np.argmax(rot, axis=1)
# print("pred rot ", rot[0])
# rot = rot[0] * np.pi / 30


# pred_img = frame.copy()
# pred_pose = obj.label2pose(viewpt, -rot, offset, depth)

# obj.setModelviewMatrix(pred_pose)
# obj.findVisibleSamplePoint()
# for pt in obj.sharp_2d_pts:
#     pt = mapt(int, pt)
#     pred_img = cv2.circle(pred_img, pt, radius=0, color=(0, 255, 0), thickness=-1)

# originPoint = mapt(int, obj.project3Dto2D((0.0, 0.0, 0.0), pred_pose))
# xaxis = mapt(int, obj.project3Dto2D((0.05, 0.0, 0.0), pred_pose))
# pred_img = cv2.line(pred_img, originPoint, xaxis, (255, 0, 0), 1)
# yaxis = mapt(int, obj.project3Dto2D((0.0, 0.05, 0.0), pred_pose))
# pred_img = cv2.line(pred_img, originPoint, yaxis, (0, 255, 0), 1)
# zaxis = mapt(int, obj.project3Dto2D((0.0, 0.0, 0.05), pred_pose))
# pred_img = cv2.line(pred_img, originPoint, zaxis, (0, 0, 255), 1)

# cv2.imshow("pred", pred_img)
# cv2.waitKey(0)
