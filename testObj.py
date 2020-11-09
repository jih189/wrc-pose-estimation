import src.common.object_model as OM
import src.configuration as CFG
import time
import cv2
import numpy as np


def obj_init():
    OM.setup(CFG.CAMERA_W, CFG.CAMERA_H)
    OM.setProjectMatrixWithIntr(CFG.CAMERA_MATRIX, CFG.CAMERA_W, CFG.CAMERA_H)

    obj = OM.ObjectModel()
    obj.loadObjectCADModel(CFG.CAD_MODEL)
    obj.setIntrinsicMatrix(CFG.CAMERA_MATRIX)

    obj.determineSharpEdges(0.8)
    obj.generateSamplePoints(0.00001, 0.00001)
    return obj


imgname = "data/images/pulley-test/000000.png"
posename = "data/images/pulley-test/000000.npy"
img = cv2.imread(imgname)
initpose = np.load(posename)

obj = obj_init()
# set pose on object
obj.setModelviewMatrix(initpose)

findVisibleSamplePoints_start = time.time()
# generate edge of on the object
obj.findVisibleSamplePoint_test()
findVisibleSamplePoints_end = time.time()
print(
    "find visible sample points time = ",
    findVisibleSamplePoints_end - findVisibleSamplePoints_start,
)
print("len of sharp_sample_points = ", len(obj.sharp_sample_points))

getPointCloud_start = time.time()
obj.getVisiblePointCloud_test()
getPointCloud_end = time.time()
print("get visible points cloud time = ", getPointCloud_end - getPointCloud_start)
# getEdge_start = time.time()
# edge = obj.getEdge(img.shape[0], img.shape[1])
# getEdge_end = time.time()
# print("get edge time = ", getEdge_end - getEdge_start)
