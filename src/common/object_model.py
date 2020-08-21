import pywavefront
from pywavefront import visualization, Wavefront
import pygame
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.ARB.occlusion_query import *
import time
import cv2
import math
from scipy.spatial.transform import Rotation as R
import torch
from torch.nn import functional as F

import sys

# map the pose to the pose which has the same shape, so it can avoid the symetric issue
def symmetricRemove(pose_input):
    eulerVec = (R.from_matrix(pose_input[:3, :3])).as_euler("ZYX")
    eulerVec[2] = 0
    pose_input[:3, :3] = (R.from_euler("ZYX", eulerVec)).as_matrix()
    return pose_input


def symmetricRemove_housing(pose_input):
    eulerVec = (R.from_matrix(pose_input[:3, :3])).as_euler("ZYX")
    eulerVec[2] = eulerVec[2] % (np.pi / 2)
    pose_input[:3, :3] = (R.from_euler("ZYX", eulerVec)).as_matrix()
    return pose_input


def py_ang(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)


# fibonacci sphere points generator
def fibonacci_sphere(samples=1):

    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points


# generate random direction
def directionGen(samples):
    # generate fibonacci points around the object for view
    points = fibonacci_sphere(samples)
    # generate the euler rotation XYZ(ext)
    direction = []
    directionPoints = []
    for p in points:
        X = 0.0
        Y = math.atan2(-p[0], p[2])
        Z = math.atan2(p[1], np.sqrt(p[0] * p[0] + p[2] * p[2]))

        direction.append([Z, Y, X])
        directionPoints.append(p)
    return np.array(direction), np.array(directionPoints)


# convert view point to view point index
def cal_idx(viewpoint):
    direction = fibonacci_sphere(64)
    viewpoint *= -1
    differenceList = []
    for v in range(len(direction)):
        differenceList.append(py_ang(viewpoint, np.array(direction[v])))
    matchindex = np.argmin(differenceList)

    return matchindex


def idx2vp(idx):
    direction = fibonacci_sphere(64)
    return (-direction[idx][0], -direction[idx][1], -direction[idx][2])


class ObjectModel:
    def __init__(self):
        self.mesh_obj = None
        self.sharp_edges = []
        self.sharp_sample_points = []
        self.sharp_sample_points_edge_indices = []
        self.visible_sharpedge_samplepoint = []
        self.visible_sharpedge_samplepoint_edge_membership = []
        self.dl = 0
        self.intrinsic = None
        self.pose = None
        self.sharp_2d_pts = []
        self.templateKernel = None
        self.kernelSize = None
        self.pointcloud = []
        self.height = 480
        self.width = 640

    # project a 3d point to a 2d point with pose
    # input: (3,) numpy matrix
    # output: (2,) numpy matrix
    def project3Dto2D(self, pt3, pose):
        if type(pt3) != tuple:
            raise Exception("Error: 3d point should be tuple ", pt3)
        # convert to numpy
        pt3 = np.array([[pt3[0], pt3[1], pt3[2], 1]]).T
        pt3_cam = (np.dot(pose, pt3))[:3, 0]
        fx = self.intrinsic[0, 0]
        fy = self.intrinsic[1, 1]
        ux = self.intrinsic[0, 2]
        uy = self.intrinsic[1, 2]
        return (
            (fx * pt3_cam[0] / pt3_cam[2] + ux),
            (fy * pt3_cam[1] / pt3_cam[2] + uy),
        )

    # load the object CAD model
    def loadObjectCADModel(self, file_name):
        # load the mesh data
        self.mesh_obj = pywavefront.Wavefront(file_name, collect_faces=True)
        self.dl = glGenLists(1)
        if not self.dl:
            print("Fail to create a display list")
            return
        glNewList(self.dl, GL_COMPILE)

        for mesh in self.mesh_obj.mesh_list:
            for face in mesh.faces:
                v1 = self.mesh_obj.vertices[face[0]]
                v2 = self.mesh_obj.vertices[face[1]]
                v3 = self.mesh_obj.vertices[face[2]]
                glBegin(GL_TRIANGLES)
                glVertex3f(v1[0], v1[1], v1[2])
                glVertex3f(v2[0], v2[1], v2[2])
                glVertex3f(v3[0], v3[1], v3[2])
                glEnd()
        glEndList()

    # set object pose
    def setModelviewMatrix(self, pose):
        # remove the symmetric pose
        # pose = self.symmetricRemove(pose)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glLoadMatrixf(pose.T)
        self.pose = pose.copy()
        return self.pose

    # get the pose of object
    def getModelviewMatrix(self):
        return glGetFloatv(GL_MODELVIEW_MATRIX).T

    # get the viewpoint, inplane rotation, shift to center, and depth
    def getLabel(self):

        viewpoint = self.getViewPoints(self.pose)
        # get the pose if the object is at the center of the view
        pose = np.identity(4)
        pose[:3, :3] = self.VP2Rotation(viewpoint)
        pose[2, 3] = np.linalg.norm(self.pose[:3, 3])

        currentO = self.project3Dto2D((0.0, 0.0, 0.0), self.pose)
        currenty = self.project3Dto2D((0.0, 0.01, 0.0), self.pose)

        newO = self.project3Dto2D((0.0, 0.0, 0.0), pose)
        newy = self.project3Dto2D((0.0, 0.01, 0.0), pose)

        v1 = np.array([currenty[0] - currentO[0], currenty[1] - currentO[1]])
        v2 = np.array([newy[0] - newO[0], newy[1] - newO[1]])

        uv1 = v1 / np.linalg.norm(v1)
        uv2 = v2 / np.linalg.norm(v2)
        cosang = np.dot(uv1.T, uv2)
        sinang = np.cross(uv1.T, uv2.T)
        angle = np.arctan2(sinang, cosang)

        return (
            viewpoint,
            angle,
            (currentO[0] - newO[0], currentO[1] - newO[1]),
            np.linalg.norm(self.pose[:3, 3]),
        )

    # render the mesh object
    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix()
        """
        for point in self.visible_sharpedge_samplepoint:
            glBegin(GL_POINTS)
            glVertex3f(point[0], point[1], point[2])
            glEnd()
        
        for line in self.sharp_edges:
            glBegin(GL_LINES)
            glVertex3f(line[0][0],line[0][1],line[0][2])
            glVertex3f(line[1][0],line[1][1],line[1][2])
            glEnd()
        """
        for mesh in self.mesh_obj.mesh_list:
            glBegin(GL_TRIANGLES)
            for face in mesh.faces:
                for vertex_i in face:
                    glVertex3f(*(self.mesh_obj).vertices[vertex_i])
            glEnd()

        glPopMatrix()
        glFlush()

    def determineSharpEdges(self, th_sharp):
        self.sharp_edges.clear()

        for i in range(len(self.mesh_obj.vertices)):
            vertex_association = []
            # find faces containing current vertex
            for k in range(len(self.mesh_obj.mesh_list[0].faces)):
                vi = self.mesh_obj.mesh_list[0].faces[k]
                vi = [x for x in vi]
                if (i == vi[0]) or (i == vi[1]) or (i == vi[2]):
                    for vii in range(3):
                        if vi[vii] > i:  # new vertex index should not be i or smaller
                            found = False
                            for l in range(len(vertex_association)):
                                if vertex_association[l][0] == vi[vii]:
                                    found = True
                                    if vertex_association[l][1][1] == -1:
                                        vertex_association[l][1][1] = k
                                    else:
                                        print("found a duplicated face")
                                    break
                            if not found:
                                vertex_association.append([vi[vii], [k, -1]])

            for l in range(len(vertex_association)):
                if vertex_association[l][1][1] != -1:
                    tri1 = vertex_association[l][1][0]
                    tri2 = vertex_association[l][1][1]
                    j = vertex_association[l][0]
                    # get the normal of two face
                    tripoints1 = self.mesh_obj.mesh_list[0].faces[tri1]
                    tripoints2 = self.mesh_obj.mesh_list[0].faces[tri2]
                    p0 = np.array(self.mesh_obj.vertices[tripoints1[0]])
                    p1 = np.array(self.mesh_obj.vertices[tripoints1[1]])
                    p2 = np.array(self.mesh_obj.vertices[tripoints1[2]])
                    N1 = np.cross(p1 - p0, p2 - p1)
                    N1 = N1 / N1.sum()

                    p0 = np.array(self.mesh_obj.vertices[tripoints2[0]])
                    p1 = np.array(self.mesh_obj.vertices[tripoints2[1]])
                    p2 = np.array(self.mesh_obj.vertices[tripoints2[2]])
                    N2 = np.cross(p1 - p0, p2 - p1)
                    N2 = N2 / N2.sum()
                    inner_prod = np.inner(N1, N2)
                    if -th_sharp <= inner_prod and inner_prod <= th_sharp:
                        self.sharp_edges.append(
                            (self.mesh_obj.vertices[i], self.mesh_obj.vertices[j])
                        )

    def generateSamplePoints(self, sample_step, sample_th):
        self.sharp_sample_points.clear()
        self.sharp_sample_points_edge_indices.clear()
        for l in range(len(self.sharp_edges)):
            dis = np.array(
                [
                    self.sharp_edges[l][1][0] - self.sharp_edges[l][0][0],
                    self.sharp_edges[l][1][1] - self.sharp_edges[l][0][1],
                    self.sharp_edges[l][1][2] - self.sharp_edges[l][0][2],
                ]
            )
            length = np.linalg.norm(dis)
            if length <= sample_th:
                self.sharp_sample_points.append(self.sharp_edges[l][0])
                self.sharp_sample_points_edge_indices.append(l)
                self.sharp_sample_points.append(self.sharp_edges[l][1])
                self.sharp_sample_points_edge_indices.append(l)
                continue
            u = dis / length
            step = u * sample_step
            j = 0
            while np.linalg.norm(step) * j <= length:
                self.sharp_sample_points.append(self.sharp_edges[l][0] + step * j)
                self.sharp_sample_points_edge_indices.append(l)
                j += 1

    def getCenterAngle(self, pose):
        horizontalR = -np.arctan2(pose[0, 3], pose[2, 3])
        verticalR = np.arctan2(
            pose[1, 3], np.sqrt(pose[0, 3] * pose[0, 3] + pose[2, 3] * pose[2, 3])
        )

        return horizontalR, verticalR

    def rotatePoseWithAngle(self, pose, horizontalR, verticalR):
        # r = R.from_euler('y', horizontalR)
        # Rmatrix = np.identity(4)
        # Rmatrix[:3,:3] = r.as_matrix()
        # pose = np.dot(Rmatrix, pose)

        # r = R.from_euler('x', verticalR)
        # Rmatrix = np.identity(4)
        # Rmatrix[:3,:3] = r.as_matrix()
        # pose = np.dot(Rmatrix, pose)

        Rmatrix = np.identity(4)
        Rmatrix[:3, :3] = R.from_euler("XYZ", [verticalR, horizontalR, 0]).as_matrix()
        pose = np.dot(Rmatrix, pose)

        return pose

    def getEdge(self, height, width):
        edgeImg = np.zeros((height, width), np.uint8)
        for p in self.sharp_2d_pts:
            p = (int(p[0]), int(p[1]))
            edgeImg = cv2.circle(edgeImg, p, radius=0, color=(255), thickness=-1)
        return edgeImg

    # this resample will generate random pose in roughly translation and rotation
    def resample(self, pose, numOfPose):

        # generate random value for rotation and translation
        xRot_1 = np.random.normal(0, 0.15, int(numOfPose / 2))
        yRot_1 = np.random.normal(0, 0.15, int(numOfPose / 2))
        zRot_1 = np.random.normal(0, 0.15, int(numOfPose / 2))

        xTrans_1 = np.random.normal(0, 0.03, int(numOfPose / 2))
        yTrans_1 = np.random.normal(0, 0.03, int(numOfPose / 2))
        zTrans_1 = np.random.normal(0, 0.08, int(numOfPose / 2))

        xRot_2 = np.random.normal(0, 0.3, numOfPose - int(numOfPose / 2))
        yRot_2 = np.random.normal(0, 0.3, numOfPose - int(numOfPose / 2))
        zRot_2 = np.random.normal(0, 0.3, numOfPose - int(numOfPose / 2))

        xTrans_2 = np.random.normal(0, 0.005, numOfPose - int(numOfPose / 2))
        yTrans_2 = np.random.normal(0, 0.005, numOfPose - int(numOfPose / 2))
        zTrans_2 = np.random.normal(0, 0.005, numOfPose - int(numOfPose / 2))

        xRot = np.concatenate((xRot_1, xRot_2))
        yRot = np.concatenate((yRot_1, yRot_2))
        zRot = np.concatenate((zRot_1, zRot_2))

        xTrans = np.concatenate((xTrans_1, xTrans_2))
        yTrans = np.concatenate((yTrans_1, yTrans_2))
        zTrans = np.concatenate((zTrans_1, zTrans_2))

        # rotate it to the center
        horizontalR = np.arctan2(pose[0, 3], pose[2, 3])
        r = R.from_euler("Y", -horizontalR)
        Rmatrix = np.identity(4)
        Rmatrix[:3, :3] = r.as_matrix()
        center_pose = np.dot(Rmatrix, pose)

        verticalR = np.arctan2(
            pose[1, 3], np.sqrt(pose[0, 3] * pose[0, 3] + pose[2, 3] * pose[2, 3])
        )
        r = R.from_euler("X", verticalR)
        Rmatrix[:3, :3] = r.as_matrix()
        center_pose = np.dot(Rmatrix, center_pose)

        poses = []

        for i in range(numOfPose):
            tempPose = np.identity(4)
            Rmatrix = np.identity(4)
            r = R.from_euler("X", xRot[i])
            Rmatrix[:3, :3] = r.as_matrix()
            tempPose = np.dot(Rmatrix, tempPose)

            r = R.from_euler("Y", yRot[i])
            Rmatrix[:3, :3] = r.as_matrix()
            tempPose = np.dot(Rmatrix, tempPose)

            r = R.from_euler("Z", zRot[i])
            Rmatrix[:3, :3] = r.as_matrix()
            tempPose = np.dot(Rmatrix, tempPose)

            tempPose = np.dot(center_pose, tempPose)

            tempPose[0, 3] += xTrans[i]
            tempPose[1, 3] += yTrans[i]
            tempPose[2, 3] += zTrans[i]

            # to ensure that the object doesn't go behind the camera
            tempPose[2, 3] = max(tempPose[2, 3], 0.1)

            # rotate it back
            r = R.from_euler("Y", horizontalR)
            Rmatrix[:3, :3] = r.as_matrix()
            tempPose = np.dot(Rmatrix, tempPose)

            r = R.from_euler("X", -verticalR)
            Rmatrix[:3, :3] = r.as_matrix()
            tempPose = np.dot(Rmatrix, tempPose)

            poses.append(tempPose)

        return poses

    def getVisibleArea(self):
        buffer = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        ret = np.frombuffer(buffer, dtype="ubyte").reshape(self.height, self.width, 3)
        ret = cv2.flip(ret, 0)
        return ret[:, :, 0]

    def getVisiblePointCloud(self):
        self.pointcloud.clear()
        model = glGetDoublev(GL_MODELVIEW_MATRIX)
        proj = glGetDoublev(GL_PROJECTION_MATRIX)
        view = glGetIntegerv(GL_VIEWPORT)
        z = glReadPixels(0, 0, self.width, self.height, GL_DEPTH_COMPONENT, GL_FLOAT)

        z = np.frombuffer(z, np.float32).reshape(self.height, self.width, 1)

        result = np.zeros((self.height, self.width, 3))  # x y z
        for x in range(self.width):
            for y in range(self.height):
                x3d, y3d, z3d = gluUnProject(x, y, z[y, x], model, proj, view)
                result[y, x, 0] = x3d
                result[y, x, 1] = y3d
                result[y, x, 2] = z3d

        result = cv2.flip(result, 0)
        for x in range(self.width):
            for y in range(self.height):
                if (
                    result[y, x, 0] > -3.0
                    and result[y, x, 0] < 3.0
                    and result[y, x, 1] > -3.0
                    and result[y, x, 1] < 3.0
                    and result[y, x, 2] > -3.0
                    and result[y, x, 2] < 3.0
                ):
                    self.pointcloud.append(
                        (y, x, result[y, x, 0], result[y, x, 1], result[y, x, 2])
                    )

    def getOptFlowWithPoses(self, height, width, targetpose):

        # do not need if object is already rendered
        # self.setModelviewMatrix(initpose)
        # self.findVisibleSamplePoint()

        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.getVisiblePointCloud()
        for i in range(len(self.pointcloud)):
            y2d, x2d, x3d, y3d, z3d = self.pointcloud[i]
            (xn, yn) = self.project3Dto2D((x3d, y3d, z3d), targetpose)
            img[int(y2d), int(x2d), 0] = int(((xn - x2d) / width + 0.5) * 255)
            img[int(y2d), int(x2d), 1] = int(((yn - y2d) / height + 0.5) * 255)

        return img

    def findVisibleSamplePoint(self):
        boundingtopleft, boundingbuttonright = None, None
        glPushMatrix()

        # disable writing to depth buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Draw the face (fill) with offset
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)
        glColor3f(1.0, 1.0, 1.0)

        # draw object model saved in display list
        glCallList(self.dl)

        glDisable(GL_POLYGON_OFFSET_FILL)

        # Occlusion test
        N = len(self.sharp_sample_points)  # number of test points
        if N <= 0:
            print("no sample points!!!")
            return

        # create a query
        vQueries = glGenQueriesARB(N)
        # Turn on occlusion testing
        # disable rendering to screen (set the color mask of all channels to False)
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
        glDepthMask(GL_FALSE)
        glPointSize(1)

        k = 0
        i = 0
        while i < N:
            glBeginQueryARB(GL_SAMPLES_PASSED_ARB, vQueries[k])
            k += 1
            glBegin(GL_POINTS)
            glVertex3f(
                self.sharp_sample_points[i][0],
                self.sharp_sample_points[i][1],
                self.sharp_sample_points[i][2],
            )
            glEnd()
            glEndQueryARB(GL_SAMPLES_PASSED_ARB)
            i += 1

        i = int(N * 3 / 4)

        ready = 0
        while not ready:
            ready = glGetQueryObjectivARB(vQueries[i], GL_QUERY_RESULT_AVAILABLE_ARB)

        # turn off occlusion testing
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
        glDepthMask(GL_TRUE)

        k = 0  # start index

        self.visible_sharpedge_samplepoint.clear()
        self.visible_sharpedge_samplepoint_edge_membership.clear()

        # clear sharp 2d points
        self.sharp_2d_pts.clear()
        for i in range(N):
            passed = glGetQueryObjectuivARB(vQueries[k], GL_QUERY_RESULT_ARB)
            k += 1
            if passed:
                self.visible_sharpedge_samplepoint.append(self.sharp_sample_points[i])
                self.visible_sharpedge_samplepoint_edge_membership.append(
                    self.sharp_sample_points_edge_indices[i]
                )
                pt2 = self.project3Dto2D(tuple(self.sharp_sample_points[i]), self.pose)
                self.sharp_2d_pts.append(pt2)
                if boundingbuttonright == None:
                    boundingbuttonright = list(pt2)
                    boundingtopleft = list(pt2)
                else:
                    if pt2[0] < boundingtopleft[0]:
                        boundingtopleft[0] = pt2[0]
                    if pt2[1] < boundingtopleft[1]:
                        boundingtopleft[1] = pt2[1]
                    if pt2[0] > boundingbuttonright[0]:
                        boundingbuttonright[0] = pt2[0]
                    if pt2[1] > boundingbuttonright[1]:
                        boundingbuttonright[1] = pt2[1]
        glDeleteQueriesARB(vQueries)
        glPopMatrix()
        return boundingtopleft, boundingbuttonright

    def setIntrinsicMatrix(self, intrinsic_):
        self.intrinsic = intrinsic_.copy()

    def compDT(self, img, isBlackBackground):
        # if it is black background, then it needs to flip
        img_temp = img.copy()
        if isBlackBackground:
            img_temp = cv2.bitwise_not(img_temp)
        result = cv2.distanceTransform(img_temp, cv2.DIST_L2, 5)
        return result

    def rotate(self, origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return int(qx), int(qy)

    def rotateImg(self, input, angle):
        _, height, width = input.shape
        result = np.zeros((height, width), np.float32)
        centery = height / 2
        centerx = width / 2
        for h in range(height):
            for w in range(width):
                # todo
                n_h, n_w = self.rotate((centery, centerx), (h, w), angle)
                if n_h < 0 or n_w < 0 or n_h >= height or n_w >= width:
                    continue
                elif input[0, h, w] != 0.0:
                    result[n_h, n_w] = input[0, h, w]
        return result

    # generate the kenerl of the object
    def generateKernel(self):
        numOfView = 64
        angleSteps = 120
        directionrpy, direction3d = directionGen(numOfView)
        Cur_matrix = np.identity(4)
        Cur_matrix[2, 3] = 1.0
        views = []

        maxsize = [0, 0]

        start = time.time()
        # generate different pose of the object and draw the edge in the views as the kernel
        for v in range(numOfView):
            # generate the pose
            curpose = (R.from_euler("XYZ", directionrpy[v])).as_matrix()
            Cur_matrix[:3, :3] = curpose
            self.setModelviewMatrix(Cur_matrix)
            upperleft, lowerright = self.findVisibleSamplePoint()
            painter = np.zeros(
                (
                    int(lowerright[1] - upperleft[1]) + 4,
                    int(lowerright[0] - upperleft[0]) + 4,
                ),
                np.uint8,
            )

            # draw the edge of the object on the painter
            for p in self.sharp_2d_pts:
                pt = (p[0] - upperleft[0] + 2, p[1] - upperleft[1] + 2)
                painter = cv2.circle(painter, pt, radius=0, color=(255), thickness=-1)
            cv2.imwrite(str(p) + ".png", painter)
            # record the max size for all kernel
            maxsize = [
                max(maxsize[0], painter.shape[0]),
                max(maxsize[1], painter.shape[1]),
            ]
            views.append(painter.copy())
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        end = time.time()
        print("time of generate view points = ", end - start)

        m_d = max(maxsize[0], maxsize[1])
        maxsize = [m_d, m_d]

        # for each views, generate all edge for different angles
        start = time.time()
        view_tmp_list = []
        for v in range(len(views)):
            view_tmp = np.zeros((maxsize[0], maxsize[1]), np.float32)
            l_row = int((maxsize[0] - views[v].shape[0]) / 2)
            l_col = int((maxsize[1] - views[v].shape[1]) / 2)
            view_tmp[
                l_row : l_row + views[v].shape[0], l_col : l_col + views[v].shape[1]
            ] = views[v]
            views[v] = np.array([view_tmp]) / 255.0
            for an in range(angleSteps):
                tmp = self.rotateImg(views[v], 3.14 / 60 * an)
                view_tmp_list.append([tmp])
        views = np.asarray(views)
        views = np.asarray(view_tmp_list)
        end = time.time()
        print(
            "time of generate in plane rotations for each view points = ", end - start
        )
        print("kernel size = ", maxsize)
        self.kernelSize = maxsize
        self.templateKernel = torch.from_numpy(views)

    # convert view point, inplane rotation, offset, and depth to pose
    def label2pose(self, viewpoint, inplaneR, offset, depth):
        pose = np.identity(4)
        pose[:3, :3] = self.VP2Rotation(viewpoint)
        pose[2, 3] = depth
        r = R.from_euler("Z", -inplaneR)
        pose[:3, :3] = np.dot(r.as_matrix(), pose[:3, :3])

        horizontalR = np.arctan2(offset[0], self.intrinsic[0, 0])
        r = R.from_euler("Y", horizontalR)
        Rmatrix = np.identity(4)
        Rmatrix[:3, :3] = r.as_matrix()
        pose = np.dot(Rmatrix, pose)
        verticalR = np.arctan2(
            offset[1],
            np.sqrt(offset[0] * offset[0] + self.intrinsic[0, 0] * self.intrinsic[0, 0])
            * self.intrinsic[1, 1]
            / self.intrinsic[0, 0],
        )
        r = R.from_euler("X", -verticalR)
        Rmatrix = np.identity(4)
        Rmatrix[:3, :3] = r.as_matrix()
        pose = np.dot(Rmatrix, pose)

        return pose

    # use the kernel of the object to compare the edge of the image
    def analysizePose(self, image):
        image = cv2.resize(
            image,
            (int(self.kernelSize[0]), int(self.kernelSize[1])),
            interpolation=cv2.INTER_AREA,
        )
        m_d = int(max(image.shape[0], image.shape[1]) * 1.2)
        print(image.shape)
        print(self.kernelSize)

        view_tmp = np.zeros((m_d, m_d), np.uint8)
        l_row = int((m_d - image.shape[0]) / 2)
        l_col = int((m_d - image.shape[1]) / 2)
        view_tmp[l_row : l_row + image.shape[0], l_col : l_col + image.shape[1]] = image
        cv2.imshow("input", view_tmp)
        view_tmp = self.compDT(view_tmp, True)
        img = torch.from_numpy(view_tmp)
        img.unsqueeze_(0)
        img.unsqueeze_(0)
        numOfView = 64
        angleSteps = 120
        similarTh = 90.0
        result = F.conv2d(img.cuda(), self.templateKernel.cuda())
        result = result.cpu()
        # result is [batch size, number of view * angles, width * height]
        result = result.view(1, numOfView * angleSteps, -1)
        # find the min cost for each views
        min_result = torch.min(result, 2)
        # arg_min_result = torch.min(min_result.values)
        numpyResult = min_result.values.numpy()

        numpyResult[numpyResult < similarTh] = 0.0
        numpyResult[numpyResult >= similarTh] = 1.0
        numpyResult = np.reshape(numpyResult, (-1, angleSteps))
        numpyResult = np.all(numpyResult, axis=1)
        print(numpyResult)

    # convert pose to view point
    def getViewPoints(self, pose):
        camera_pose = np.linalg.inv(pose)
        tx = camera_pose[0, 3] * -1
        ty = camera_pose[1, 3] * -1
        tz = camera_pose[2, 3] * -1
        res = np.array([tx, ty, tz])
        res = res / np.linalg.norm(res)
        return res

    def VP2Rotation(self, p):
        X = 0.0
        Y = math.atan2(-p[0], p[2])
        Z = math.atan2(p[1], np.sqrt(p[0] * p[0] + p[2] * p[2]))

        rot_m = np.identity(3)
        rot_m = R.from_euler("XYZ", [Z, Y, X]).as_matrix()
        return rot_m

    def symmetricAnalysis(self):
        numOfView = 26  # number of different views of the object
        angleSteps = 120  # number of step angle rotation
        similarTh = 100.0  # similarity threhold
        directionrpy, direction3d = directionGen(numOfView)
        Cur_matrix = np.identity(4)
        Cur_matrix[2, 3] = 0.8
        views = []

        maxsize = [0, 0]

        # generate different view points for the object
        start = time.time()
        for v in range(numOfView):
            curpose = (R.from_euler("XYZ", directionrpy[v])).as_matrix()
            Cur_matrix[:3, :3] = curpose
            self.setModelviewMatrix(Cur_matrix)
            upperleft, lowerright = self.findVisibleSamplePoint()
            painter = np.zeros(
                (
                    int(lowerright[1] - upperleft[1]) + 4,
                    int(lowerright[0] - upperleft[0]) + 4,
                ),
                np.uint8,
            )

            # draw current view point
            for p in self.sharp_2d_pts:
                pt = (p[0] - upperleft[0] + 2, p[1] - upperleft[1] + 2)
                painter = cv2.circle(painter, pt, radius=0, color=(255), thickness=-1)
            # save the view point
            cv2.imwrite("view" + str(v) + ".png", painter)
            maxsize = [
                max(maxsize[0], painter.shape[0]),
                max(maxsize[1], painter.shape[1]),
            ]
            views.append(painter.copy())
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        end = time.time()
        print("time of generate view points = ", end - start)

        # get the max size of all view point
        m_d = max(maxsize[0], maxsize[1])
        maxsize = [m_d, m_d]

        start = time.time()
        largerView = []
        view_tmp_list = []
        # generate different rotations view for earch view point
        for v in range(len(views)):
            largerView_tmp = np.zeros((maxsize[0] * 2, maxsize[1] * 2), np.uint8)
            l_row = int((maxsize[0] * 2 - views[v].shape[0]) / 2)
            l_col = int((maxsize[1] * 2 - views[v].shape[1]) / 2)
            # create the larger image for earch view points for comparing with the kernel later
            largerView_tmp[
                l_row : l_row + views[v].shape[0], l_col : l_col + views[v].shape[1]
            ] = views[v]
            largerView_tmp = self.compDT(largerView_tmp, True)
            largerView.append(largerView_tmp)
            view_tmp = np.zeros((maxsize[0], maxsize[1]), np.float32)
            l_row = int((maxsize[0] - views[v].shape[0]) / 2)
            l_col = int((maxsize[1] - views[v].shape[1]) / 2)
            view_tmp[
                l_row : l_row + views[v].shape[0], l_col : l_col + views[v].shape[1]
            ] = views[v]
            views[v] = np.array([view_tmp]) / 255.0
            for an in range(angleSteps):
                tmp = self.rotateImg(views[v], 3.14 / 60 * an)
                view_tmp_list.append([tmp])
        views = np.asarray(views)
        views = np.asarray(view_tmp_list)
        end = time.time()
        print(
            "time of generate in plane rotations for each view points = ", end - start
        )

        # compare each pair of view points
        start = time.time()
        views = torch.from_numpy(views)
        similarView = []
        for v in range(len(largerView)):
            largerView[v] = torch.from_numpy(largerView[v])
            largerView[v].unsqueeze_(0)
            largerView[v].unsqueeze_(0)
            result = F.conv2d(largerView[v].cuda(), views.cuda())
            result = result.cpu()
            result = result.view(1, numOfView * angleSteps, -1)
            min_result = torch.min(result, 2)
            # arg_min_result = torch.min(min_result.values)
            numpyResult = min_result.values.numpy()

            numpyResult[numpyResult < similarTh] = 0.0
            numpyResult[numpyResult >= similarTh] = 1.0
            numpyResult = np.reshape(numpyResult, (-1, angleSteps))
            numpyResult = np.all(numpyResult, axis=1)
            similarView.append(numpyResult)
        end = time.time()
        print("time of compare each pairs of them = ", end - start)

        # group the views which are similar
        group = {}
        for i in range(numOfView):
            group[i] = -1

        for i in range(numOfView):
            for j in range(i + 1, numOfView):
                if similarView[i][j] == False and similarView[j][i] == False:
                    if group[i] == -1:
                        if group[j] == -1:
                            group[j] = i
                        elif group[j] != i:
                            group[i] = group[j]
                            # print("confliction from ", j, " to ", i, " and ", group[j])
                    else:
                        if group[j] == -1:
                            group[j] = group[i]
                        elif group[j] != group[i]:
                            print("confliction from ", j, " to ", i, " and ", group[j])

        setGroup = {}
        for i in range(numOfView):
            if group[i] == -1:
                setGroup[i] = []
                setGroup[i].append(i)
            else:
                if not group[i] in setGroup.keys():
                    setGroup[group[i]] = []
                setGroup[group[i]].append(i)

        for s in setGroup:
            print(setGroup[s])


def testInPygame():
    pygame.display.flip()
    pygame.time.wait(1000)


# setup the pygame
def setup(width, height):
    pygame.init()
    pygame.display.set_mode((width, height), pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("Test demo")
    pygame.display.iconify()

    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_FLAT)


# set camera intrinsic matrix
def setProjectMatrixWithIntr(intrinsic, width, height):

    glMatrixMode(GL_PROJECTION)
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    ux = intrinsic[0, 2]
    uy = intrinsic[1, 2]
    width_ = width
    height_ = height

    near = 0.1
    far = 30.0

    Mp = np.array(
        [
            [2.0 * fx / width_, 0.0, 0.0, 0.0],
            [0, -2.0 * fy / height_, 0.0, 0.0],
            [
                2.0 * ux / width_ - 1.0,
                -2.0 * uy / height_ + 1.0,
                (far + near) / (far - near),
                1.0,
            ],
            [0.0, 0.0, -2.0 * far * near / (far - near), 0.0],
        ]
    )

    glLoadMatrixf(Mp)
