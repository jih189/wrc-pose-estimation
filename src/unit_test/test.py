# this is a script to test functions
import unittest
import src.common.object_model as OM
import src.configuration as CFG
import numpy as np
import cv2
from numpy.testing import assert_array_almost_equal

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchgeometry as tgm
import kornia

# ignore warming
np.seterr(divide="ignore", invalid="ignore")


class test_object_model(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # load the object mesh
        OM.setup(CFG.CAMERA_W, CFG.CAMERA_H)
        OM.setProjectMatrixWithIntr(CFG.CAMERA_MATRIX, CFG.CAMERA_W, CFG.CAMERA_H)

        cls.obj = OM.ObjectModel()
        cls.obj.setIntrinsicMatrix(CFG.CAMERA_MATRIX)
        cls.obj.loadObjectCADModel(CFG.CAD_MODEL)

        cls.obj.determineSharpEdges(0.05)
        cls.obj.generateSamplePoints(0.0001, 0.001)

    #
    def test_project3Dto2D(self):
        pose = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.1],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        input = (0, 0, 0)
        output = (CFG.CAMERA_MATRIX[0, 2], CFG.CAMERA_MATRIX[1, 2])
        result = self.obj.project3Dto2D(input, pose)
        assert_array_almost_equal(output, result)

    def test_getLabel(self):
        pose = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.1],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.obj.setModelviewMatrix(pose)
        viewpoint, angle, offset, distance = self.obj.getLabel()
        self.assertEqual(distance, 0.1)

    def test_move_to_center(self):
        img = cv2.imread("src/unit_test/image.png")
        pose = np.load("src/unit_test/pose.npy")

        self.obj.setModelviewMatrix(pose)
        self.obj.findVisibleSamplePoint()

        for p in self.obj.sharp_2d_pts:
            img = cv2.circle(
                img, (int(p[0]), int(p[1])), radius=0, color=(0, 255, 0), thickness=-1,
            )

        target3dPts = np.array(self.obj.visible_sharpedge_samplepoint)
        target3dPt = Variable(torch.from_numpy(target3dPts)).float()
        target3dPt = target3dPt.unsqueeze(0)

        # generate set of random poses
        random_poses = self.obj.resample(pose, 1)

        # mask bit
        target_mask = self.obj.getVisibleArea(img)

        for random_pose in random_poses:
            # get current pose if it is moved to the center
            horizontalR, verticalR = self.obj.getCenterAngle(random_pose)

            # set pose on object
            self.obj.setModelviewMatrix(random_pose)

            # generate edge of on the object
            self.obj.findVisibleSamplePoint()

            # generate preprocessed data
            # inital pose mask
            mask = self.obj.getVisibleArea(img)

            # find the crop size
            [_, _, w, h] = cv2.boundingRect(mask)

            boundingsize = max(w, h) * 2.0

            # get center point from pose
            centerPoint = self.obj.project3Dto2D((0, 0, 0), random_pose)

            ex = int(centerPoint[0] - boundingsize / 2)  # here is wrong
            ey = int(centerPoint[1] - boundingsize / 2)
            ew = int(boundingsize)
            eh = int(boundingsize)

            if ew == 0 or eh == 0:
                logger.warn(f"Invalid image width/height. Continuing...")
                continue

            if ex < 0 or ey < 0 or ex + ew >= img.shape[1] or ey + eh >= img.shape[0]:
                logger.warn(f"Bounding box out of image. Continuing...")
                continue

            # cropped image with initial pose as center
            crop_img = img[ey : ey + eh, ex : ex + ew]

            # apply same rotation as above
            target_pose_at_center = self.obj.rotatePoseWithAngle(
                pose, horizontalR, verticalR
            )

            targetPose = Variable(torch.from_numpy(target_pose_at_center)).float()
            targetPose = targetPose.unsqueeze(0)

            camera_matrix = torch.tensor(
                [
                    [
                        [CFG.CAMERA_MATRIX[0, 0], 0.0, float(crop_img.shape[0]) / 2,],
                        [0.0, CFG.CAMERA_MATRIX[1, 1], float(crop_img.shape[1]) / 2,],
                        [0.0, 0.0, 1.0],
                    ]
                ]
            )

            label3dpt = tgm.transform_points(targetPose, target3dPt)
            label_2d_pts = kornia.project_points(label3dpt, camera_matrix)

            for p in label_2d_pts.cpu().detach().numpy()[0]:
                crop_img = cv2.circle(
                    crop_img,
                    (int(p[0]), int(p[1])),
                    radius=0,
                    color=(0, 0, 255),
                    thickness=-1,
                )

            # cv2.imshow("test", crop_img)
            # cv2.waitKey(0)
            # print("before destory windows")

            # cv2.destroyAllWindows()
            print("done")

    @classmethod
    def tearDownClass(cls):
        print("tear down class")


if __name__ == "__main__":
    unittest.main()
