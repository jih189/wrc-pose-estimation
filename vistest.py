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

rotation = 0
window = OM.setup(CFG.CAMERA_W, CFG.CAMERA_H)
OM.setProjectMatrixWithIntr(CFG.CAMERA_MATRIX, CFG.CAMERA_W, CFG.CAMERA_H)

obj = OM.ObjectModel()
obj.setIntrinsicMatrix(CFG.CAMERA_MATRIX)
obj.loadObjectCADModel(CFG.SAMPLE_FACE_MODEL)

# meshes = pywavefront.Wavefront(CFG.SAMPLE_FACE_MODEL, collect_faces=True)

pose = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.3],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
obj.setModelviewMatrix(pose)
print("window stype")
print(type(window))
while True:
    glRotatef(5.0, 1, 0, 0)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                glTranslatef(-0.01, 0.0, 0.0)
            if event.key == pygame.K_RIGHT:
                glTranslatef(0.01, 0.0, 0.0)
            if event.key == pygame.K_UP:
                glTranslatef(0.0, 0.0, 0.01)
            if event.key == pygame.K_DOWN:
                glTranslatef(0.0, 0.0, -0.01)
    obj.render()
    img = pygame.image.tostring(window, "RGB", False)
    img = np.fromstring(img, np.uint8).reshape(CFG.CAMERA_H, CFG.CAMERA_W, 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow("img", img)
    if cv2.waitKey(3) & 0xFF == ord("q"):
        break
