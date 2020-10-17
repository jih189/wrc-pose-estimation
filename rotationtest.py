import cv2
from scipy.io import loadmat
import src.common.object_model as OM
import numpy as np
import tqdm
from scipy.spatial.transform import Rotation as R
import src.configuration as CFG


def init():
    # load the object mesh
    OM.setup(CFG.CAMERA_W, CFG.CAMERA_H)
    OM.setProjectMatrixWithIntr(CFG.CAMERA_MATRIX, CFG.CAMERA_W, CFG.CAMERA_H)
    obj = OM.ObjectModel()
    obj.setIntrinsicMatrix(CFG.CAMERA_MATRIX)
    obj.loadObjectCADModel(CFG.CAD_MODEL)

    obj.determineSharpEdges(0.8)
    obj.generateSamplePoints(0.001, 0.0001)

    return obj


def rotate_image(image, angle, rotate_center):
    rot_mat = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def get_centered_crop(topleft, botright):
    cropHeight = botright[1] - topleft[1]
    cropWidth = botright[0] - topleft[0]

    centerPoint = (topleft[0] + cropWidth / 2, topleft[1] + cropHeight / 2)

    cropSize = max(cropHeight, cropWidth)

    topleft_new = np.array(
        [centerPoint[0] - cropSize / 2, centerPoint[1] - cropSize / 2], dtype=int
    )
    botright_new = np.array(
        [centerPoint[0] + cropSize / 2, centerPoint[1] + cropSize / 2], dtype=int
    )

    return topleft_new, botright_new


def rotateAngle(pose, angle):
    angle = np.radians(angle)
    horizontalR = -np.arctan2(pose[0, 3], pose[2, 3])
    verticalR = np.arctan2(
        pose[1, 3], np.sqrt(pose[0, 3] * pose[0, 3] + pose[2, 3] * pose[2, 3])
    )

    Rmatrix = np.identity(4)
    Rmatrix[:3, :3] = R.from_euler("XYZ", [verticalR, horizontalR, 0]).as_matrix()
    pose = np.dot(Rmatrix, pose)

    Rmatrix[:3, :3] = R.from_euler("Z", -angle).as_matrix()

    pose = np.dot(Rmatrix, pose)
    Rmatrix[:3, :3] = R.from_euler("XYZ", [-verticalR, -horizontalR, 0]).as_matrix()
    pose = np.dot(Rmatrix, pose)
    return pose


if __name__ == "__main__":
    rotateDegree = 32.0
    obj = init()

    img = cv2.imread("test.png")
    img = np.array(img)
    demo_img = img.copy()

    pose = np.load("test.npy")
    obj.setModelviewMatrix(pose)
    obj.findVisibleSamplePoint()
    for p in obj.sharp_2d_pts:
        p = (int(p[0]), int(p[1]))
        demo_img = cv2.circle(demo_img, p, radius=0, color=(0, 0, 255), thickness=-1)

    center_pt = obj.project3Dto2D((0, 0, 0), pose)
    center_pt = (int(center_pt[0]), int(center_pt[1]))

    rot_img = rotate_image(img, rotateDegree, center_pt)

    rot_pose = rotateAngle(pose, rotateDegree)
    obj.setModelviewMatrix(rot_pose)
    obj.findVisibleSamplePoint()

    for p in obj.sharp_2d_pts:
        p = (int(p[0]), int(p[1]))
        rot_img = cv2.circle(rot_img, p, radius=0, color=(0, 0, 255), thickness=-1)

    cv2.imshow("demo", demo_img)
    cv2.imshow("rot", rot_img)
    cv2.waitKey(0)

