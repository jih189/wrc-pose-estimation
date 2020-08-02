import numpy as np
import cv2
import torch
import torch.nn as nn

from torch.autograd import Variable

from models.model import Magic_Net
import src.common.object_model as OM


camera_matrix = np.array(
    [
        [654.968116289191, 0, 322.67377109101744],
        [0, 657.1436336052552, 248.70937432215163],
        [0, 0, 1],
    ],
    dtype="double",
)


OM.setup(640, 480)

OM.setProjectMatrixWithIntr(camera_matrix, 640, 480)

obj = OM.ObjectModel()
obj.loadObjectCADModel("MBRFA30-2-P6.obj")
obj.setIntrinsicMatrix(camera_matrix)

obj.determineSharpEdges(0.05)
obj.generateSamplePoints(0.001, 0.001)

test_file = "002450"

img_path = "data/processed/pulley_crop/crop" + test_file + ".png"
# img_path = "test0.png"
img = cv2.imread(img_path)
img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_AREA)
img = img[:, :, :3].transpose(2, 0, 1)
img = img[np.newaxis, ...]

input = Variable(torch.from_numpy(img).cuda()).float()
# print(input.shape)

viewpt_class = 64
rot_class = 60

model = Magic_Net(viewpt_class=viewpt_class, rot_class=rot_class).cuda()
model = nn.DataParallel(model)
model = torch.load("best_model_rot.pth")
model.eval()

output = model(input)

pred = output[:, :viewpt_class].data.cpu().numpy()
pred = np.argmax(pred, axis=1)
print("viewpoint index = ", pred[0])
viewpt = np.array(OM.idx2vp(pred[0]))

rot = output[:, viewpt_class : viewpt_class + rot_class].data.cpu().numpy()
value, idx = output[:, viewpt_class : viewpt_class + rot_class].topk(
    5, 1, largest=True, sorted=True
)
idx = idx[0].data.cpu().numpy()
rot = np.argmax(rot, axis=1)
print("rot class index = ", rot[0])
rot = rot[0] * np.pi / 30

boundingbox = np.load("data/processed/pulley_crop/bounding" + test_file + ".npy")
# boundingbox =np.load('bounding0.npy')

upperleftx, upperlefty, lowerrightx, lowerrighty = (
    boundingbox[0].astype(np.int),
    boundingbox[1].astype(np.int),
    boundingbox[2].astype(np.int),
    boundingbox[3].astype(np.int),
)
l = lowerrightx - upperleftx
principle_pt = np.array([322.673, 248.709])

position = torch.sigmoid(output[:, viewpt_class + rot_class :]).data.cpu().numpy()
position *= l
offset = position[:, :2]
# offset = np.array([0.5,0.5])
# offset *= l

offset = np.array([upperleftx, upperlefty]) + offset.reshape(2) - principle_pt


pose_label = np.load("data/processed/pulley_crop/" + test_file + ".npy")
depth = np.linalg.norm(pose_label[:3, 3])

pose = obj.label2pose(viewpt, rot, offset, depth)
# print(pose)

frame = cv2.imread("data/raw/pulley/" + test_file + ".png")
# frame = cv2.imread("ori0.png")
# frame = cv2.circle(frame, (upperleftx, upperlefty), radius=2, color=(0,0,255), thickness=-1)
# frame = cv2.circle(frame, (lowerrightx, lowerrighty), radius=2, color=(0,0,255), thickness=-1)
# viewWindow = np.zeros(frame.shape,np.uint8)
# pose = np.identity(4)
# pose[2,3] = 0.5
# display coordinate


originPoint = obj.project3Dto2D((0.0, 0.0, 0.0), pose)
xaxis = obj.project3Dto2D((0.05, 0.0, 0.0), pose)
frame = cv2.line(frame, originPoint, xaxis, (255, 0, 0), 1)
yaxis = obj.project3Dto2D((0.0, 0.05, 0.0), pose)
frame = cv2.line(frame, originPoint, yaxis, (0, 255, 0), 1)
zaxis = obj.project3Dto2D((0.0, 0.0, 0.05), pose)
frame = cv2.line(frame, originPoint, zaxis, (0, 0, 255), 1)

obj.setModelviewMatrix(pose)
obj.findVisibleSamplePoint()
for pt in obj.sharp_2d_pts:
    frame = cv2.circle(frame, pt, radius=0, color=(0, 0, 255), thickness=-1)

obj.setModelviewMatrix(pose_label)
viewPoint, inplaneRotation, offsetFromCenter, depth = obj.getLabel()
print("label index = ", OM.cal_idx(viewPoint))
inplaneRotation = inplaneRotation % (2 * np.pi) / (2 * np.pi / 60)
inplaneRotation = int(inplaneRotation)
print("rotation label = ", inplaneRotation)
if inplaneRotation in idx:
    print("Top 5: Correct")
else:
    print("Top 5: Wrong")

# originPoint = obj.project3Dto2D((0.0,0.0,0.0), pose_label)
# xaxis = obj.project3Dto2D((0.05,0.0,0.0), pose_label)
# frame = cv2.line(frame,originPoint,xaxis,(255,0,0),1)
# yaxis = obj.project3Dto2D((0.0,0.05,0.0), pose_label)
# frame = cv2.line(frame,originPoint,yaxis,(0,255,0),1)
# zaxis = obj.project3Dto2D((0.0,0.0,0.05), pose_label)
# frame = cv2.line(frame,originPoint,zaxis,(0,0,255),1)

cv2.imshow("view", frame)
cv2.waitKey(0)
