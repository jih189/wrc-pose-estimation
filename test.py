import cv2
from scipy.io import loadmat
import src.common.object_model as OM
import numpy as np
import tqdm

MODEL_DIR = "data/external/YCB_dataset/models/"


def loadMeta(mat_file):
    # load meta data
    mat = loadmat(mat_file)
    camera_matrix = mat["intrinsic_matrix"]
    cls_indexes = mat["cls_indexes"]
    poses = mat["poses"]

    return camera_matrix, cls_indexes, poses


def init_obj(camera_h, camera_w, camera_matrix, cad_model):
    # load the object mesh
    OM.setup(camera_w, camera_h)
    OM.setProjectMatrixWithIntr(camera_matrix, camera_w, camera_h)

    obj = OM.ObjectModel()
    obj.setIntrinsicMatrix(camera_matrix)
    obj.loadObjectCADModel(cad_model)

    obj.determineSharpEdges(0.8)
    obj.generateSamplePoints(0.000001, 0.1)
    return obj


def loadClasses(classesfile):
    file1 = open(classesfile, "r")
    Lines = file1.readlines()

    count = 1
    result = []
    # Strips the newline character
    for line in Lines:
        result.append(line.strip())
        count += 1
    return result


def processPose(pose):
    pose = np.concatenate((pose, [[0, 0, 0, 1]]), 0)
    return pose


if __name__ == "__main__":

    classes = loadClasses("data/external/YCB_dataset/image_sets/classes.txt")

    targetpose = np.array(
        [
            [0.68670714, -0.7268656, -0.01003767, 0.07240387],
            [-0.39432726, -0.36086995, -0.84515083, 0.0489662],
            [0.61068822, 0.58432885, -0.53443427, 0.91424911],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # 56
    testind = 1
    for i in tqdm.tqdm(range(testind, testind + 1, 1)):
        filename = "data/external/YCB_dataset/data/0038/{:06d}".format(i)
        image_file = filename + "-color.png"
        mat_file = filename + "-meta.mat"

        # read image
        img = cv2.imread(image_file)
        height = img.shape[0]
        width = img.shape[1]

        camera_matrix, cls_indexes, poses = loadMeta(mat_file)

        for c in cls_indexes:
            # print("init", classes[c[0] - 1])
            mesh_dir = MODEL_DIR + classes[c[0] - 1] + "/textured.obj"
            # print("object mesh dir = ", mesh_dir)
            obj = init_obj(height, width, camera_matrix, mesh_dir)

            pose = processPose(poses[:, :, 0])
            # if np.allclose(pose, targetpose, rtol=0.01):
            #     print("found ", str(i))
            obj.setModelviewMatrix(pose)

            obj.findVisibleSamplePoint()
            # draw init pose
            for p in obj.sharp_2d_pts:
                img = cv2.circle(
                    img,
                    (int(p[0]), int(p[1])),
                    radius=1,
                    color=(0, 255, 0),
                    thickness=-1,
                )
            cv2.imshow("image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break

