import ctypes
import os
import sys

import pyglet
from pyglet.gl import *
import pygame

from pywavefront import visualization
import pywavefront

import src.common.object_model as OM
import src.configuration as CFG
import time
import numpy as np
import cv2

# t_LM2cv = np.array([[1,0,0],[0,1,0],[0,0,1]])
t_LM2cv = np.array([[0,-1,0],[0,0,-1],[1,0,0]])

t_cv2branch = np.array([[0,-1,0],[0,0,-1],[1,0,0]])

def linemod_pose(path, i):
    """
    read a 3x3 rotation and 3x1 translation
    @ return R, t in [m]
    """

    R = open("{}/data/rot{}.rot".format(path, i))
    R.readline()
    R = np.float32(R.read().split()).reshape((3,3))

    t = open("{}/data/tra{}.tra".format(path, i))
    t.readline()
    t = np.float32(t.read().split())

    return R, t

def lm2cv(R, t):
    R = t_LM2cv.dot(R)
    t = t_LM2cv.dot(t)

    R *= -1
    t *= -1

    return R, t

CAMERA_MATRIX = np.array(
    [[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1],], dtype="double",
)

window = OM.setup(CFG.CAMERA_W, CFG.CAMERA_H)
OM.setProjectMatrixWithIntr(CAMERA_MATRIX, CFG.CAMERA_W, CFG.CAMERA_H)

obj = OM.ObjectModel()
obj.setIntrinsicMatrix(CAMERA_MATRIX)
obj.loadObjectCADModel("data/external/LINEMOD_dataset/ape/mesh.obj")

# meshes = pywavefront.Wavefront(CFG.SAMPLE_FACE_MODEL, collect_faces=True)

# while True:
#     glRotatef(5.0, 1, 0, 0)
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#             quit()
#         if event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_LEFT:
#                 glTranslatef(-0.01, 0.0, 0.0)
#             if event.key == pygame.K_RIGHT:
#                 glTranslatef(0.01, 0.0, 0.0)
#             if event.key == pygame.K_UP:
#                 glTranslatef(0.0, 0.0, 0.01)
#             if event.key == pygame.K_DOWN:
#                 glTranslatef(0.0, 0.0, -0.01)
#     obj.render()
#     img = pygame.image.tostring(window, "RGB", False)
#     img = np.fromstring(img, np.uint8).reshape(CFG.CAMERA_H, CFG.CAMERA_W, 3)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#     cv2.imshow("img", img)
#     if cv2.waitKey(10) & 0xFF == ord("q"):
#         break


for i in range(1):
    # read image and pose
    image = cv2.imread("data/external/LINEMOD_dataset/ape/data/color" + str(i) + ".jpg")
    R, t = linemod_pose("data/external/LINEMOD_dataset/ape", i)
    pose = np.identity(4)
    t /=100.0
    pose[:3,:3] = R
    pose[:3,3] = t

    print("pose")
    print(pose)


    obj.setModelviewMatrix(pose)
    obj.render()
    img = pygame.image.tostring(window, "RGB", False)
    img = np.fromstring(img, np.uint8).reshape(CFG.CAMERA_H, CFG.CAMERA_W, 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mask = obj.getMask()
    image[mask] = image[mask] * [0,0,0]

    cv2.imshow("img", img)
    cv2.imshow("image", image)
    cv2.waitKey(0)
