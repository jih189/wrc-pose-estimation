from ar_markers import detect_markers
import src.configuration as CFG
import numpy as np
import cv2
import src.common.object_model as OM

if __name__ == "__main__":

    input_path = CFG.IMAGE_SAVE_PATH
    output_path = CFG.VERIFY_IMAGE_PATH

    # you can update this two numbers if need
    startIdx = 0
    saveInx = 0

    currentInx = startIdx

    OM.setup(CFG.CAMERA_W, CFG.CAMERA_H)

    OM.setProjectMatrixWithIntr(CFG.CAMERA_MATRIX, CFG.CAMERA_W, CFG.CAMERA_H)

    obj = OM.ObjectModel()
    obj.loadObjectCADModel(CFG.CAD_MODEL)
    obj.setIntrinsicMatrix(CFG.CAMERA_MATRIX)

    obj.determineSharpEdges(0.05)
    obj.generateSamplePoints(0.001, 0.001)

    print("s key: save the image")
    print("q key: quit")
    print("n key: next image")

    while True:
        currentName = input_path + "{:06d}".format(currentInx) + ".png"

        img = cv2.imread(currentName)

        if img.size == 0:
            print("finish!")
            break
        pose = np.load(input_path + "{:06d}".format(currentInx) + ".npy")
        obj.setModelviewMatrix(pose)

        upperleft, lowerright = obj.findVisibleSamplePoint()
        upperleft = (int(upperleft[0]), int(upperleft[1]))
        lowerright = (int(lowerright[0]), int(lowerright[1]))

        cropImg = img.copy()
        for p in obj.sharp_2d_pts:
            cropImg = cv2.circle(cropImg, p, radius=0, color=(0, 0, 255), thickness=-1)

        cropImg = cropImg[upperleft[1] : lowerright[1], upperleft[0] : lowerright[0]]
        cropImg = cv2.resize(
            cropImg,
            (cropImg.shape[0] * 10, cropImg.shape[1] * 10),
            interpolation=cv2.INTER_AREA,
        )
        cv2.imshow("crop", cropImg)

        cv2.imshow("view", img)
        ch = cv2.waitKey(0)
        if ch & 0xFF == ord("s"):  # collect data
            cv2.imwrite(output_path + "{:06d}".format(saveInx) + ".png", img)
            np.save(output_path + "{:06d}".format(saveInx) + ".npy", pose)
            saveInx += 1
            currentInx += 1
        elif ch & 0xFF == ord("q"):
            break
        elif ch & 0xFF == ord("n"):
            currentInx += 1

