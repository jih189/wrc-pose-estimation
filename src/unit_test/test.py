# this is a script to test functions
import unittest
import src.common.object_model as OM
import src.configuration as CFG
import numpy as np
from numpy.testing import assert_array_almost_equal

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

    @classmethod
    def tearDownClass(cls):
        print("tear down class")


if __name__ == "__main__":
    unittest.main()
