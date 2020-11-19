import src.configuration as CFG
import numpy as np
import cv2
import src.common.object_model as OM
from pathlib import Path

# importing shutil module
import shutil
import os

if __name__ == "__main__":

    # ensure the directory which store images and poses exists
    if not os.path.isdir(CFG.VERIFY_IMAGE_PATH):
        print("please make a directory:")
        print("mkdir -p ", CFG.VERIFY_IMAGE_PATH)
        exit()

    input_path = CFG.IMAGE_SAVE_PATH
    output_path = CFG.VERIFY_IMAGE_PATH

    OM.setup(CFG.CAMERA_W, CFG.CAMERA_H)

    OM.setProjectMatrixWithIntr(CFG.CAMERA_MATRIX, CFG.CAMERA_W, CFG.CAMERA_H)

    obj = OM.ObjectModel()
    obj.loadObjectCADModel(CFG.CAD_MODEL)
    obj.setIntrinsicMatrix(CFG.CAMERA_MATRIX)

    obj.determineSharpEdges(0.9)
    obj.generateSamplePoints(0.0001)

    print("s key: save all correct image and pose pairs")
    print("m key: next image")
    print("n key: last image")
    print("k key: keep the current pose in the frame")
    print("c key: set correct or incorrect")

    # read all pose and image names

    input_path = Path(input_path)
    image_names, pose_names = [], []
    for f in input_path.iterdir():
        if f.match("*.png"):
            image_names.append(str(f))
        if f.match("*.npy"):
            pose_names.append(str(f))
    image_names.sort()
    pose_names.sort()
    if len(image_names) == 0:
        print("there is no image and pose")
        exit()

    images_and_poses = list(zip(image_names, pose_names))

    isCorrect = [True] * len(image_names)  # bit which means the pose is correct

    currentInx = 0
    isSave = False

    while True:

        img = cv2.imread(images_and_poses[currentInx][0])
        pose = np.load(images_and_poses[currentInx][1])
        obj.setModelviewMatrix(pose)

        obj.findVisibleSamplePoint()
        bx, by, bw, bh = cv2.boundingRect(obj.getVisibleArea())
        upperleft = (bx, by)
        lowerright = (bx + bw, by + bh)

        for p in obj.sharp_2d_pts:
            p = (int(p[0]), int(p[1]))
            img = cv2.circle(img, p, radius=0, color=(0, 0, 255), thickness=-1)

        # adjust the bounding box
        crop_upperleft, crop_lowerright = OM.get_centered_crop(upperleft, lowerright)

        cropImg = np.zeros(
            (
                crop_lowerright[1] - crop_upperleft[1],
                crop_lowerright[0] - crop_upperleft[0],
                3,
            ),
            np.uint8,
        )
        upperleft_crop_inner = [
            max(0, crop_upperleft[0]),
            max(0, crop_upperleft[1]),
        ]
        lowerright_crop_inner = [
            min(img.shape[1], crop_lowerright[0]),
            min(img.shape[0], crop_lowerright[1]),
        ]
        cropImg[
            upperleft_crop_inner[1]
            - crop_upperleft[1] : lowerright_crop_inner[1]
            - crop_upperleft[1],
            upperleft_crop_inner[0]
            - crop_upperleft[0] : lowerright_crop_inner[0]
            - crop_upperleft[0],
        ] = img[
            int(upperleft_crop_inner[1]) : int(lowerright_crop_inner[1]),
            int(upperleft_crop_inner[0]) : int(lowerright_crop_inner[0]),
        ]

        cropImg = cv2.resize(cropImg, (300, 300), interpolation=cv2.INTER_AREA,)
        cv2.putText(
            img,
            "current index: " + str(currentInx),
            (0, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            img,
            "Is pose correct: " + str(isCorrect[currentInx]),
            (0, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.imshow("view", img)
        cv2.imshow("crop", cropImg)

        ch = cv2.waitKey(0)
        if ch & 0xFF == ord("c"):
            isCorrect[currentInx] = not isCorrect[currentInx]
        elif ch & 0xFF == ord("s"):  # collect data
            print("save all datas")
            isSave = True
            break
        elif ch & 0xFF == ord("q"):
            break
        elif ch & 0xFF == ord("n"):
            if currentInx > 0:
                currentInx -= 1
        elif ch & 0xFF == ord("m"):
            if currentInx < len(image_names) - 1:
                currentInx += 1

    if isSave:
        # move correct image, pose pairs to des
        saveInx = 0
        for i in range(len(images_and_poses)):
            if isCorrect[i]:
                shutil.copyfile(
                    images_and_poses[i][0],
                    output_path + "{:06d}".format(saveInx) + ".png",
                )
                shutil.copyfile(
                    images_and_poses[i][1],
                    output_path + "{:06d}".format(saveInx) + ".npy",
                )
                saveInx += 1
        print("save done!")
