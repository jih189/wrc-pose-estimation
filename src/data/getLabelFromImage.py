from __future__ import print_function
import numpy as np
import cv2
from ar_markers import detect_markers
import src.common.object_model as OM
import src.configuration as CFG

# for flat
tablePoints = [
    [-0.005, -0.0675, -0.0675],
    [-0.005, -0.0675, 0.0675],
    [-0.005, 0.0675, -0.0675],
    [-0.005, 0.0675, 0.0675],
]

# for vertical
# tablePoints = [[-0.0675, -0.0675,-0.015],
#                 [-0.0675, 0.0675,-0.015],
#                 [0.0675,-0.0675,-0.015],
#                 [0.0675, 0.0675,-0.015]]

dist_coefs = np.zeros((4, 1))

if __name__ == "__main__":
    print('Press "q" to quit')
    capture = cv2.VideoCapture(CFG.CAMERA_ID)

    if capture.isOpened():  # try to get the first frame
        frame_captured, frame = capture.read()
    else:
        frame_captured = False

    OM.setup(CFG.CAMERA_W, CFG.CAMERA_H)

    OM.setProjectMatrixWithIntr(CFG.CAMERA_MATRIX, CFG.CAMERA_W, CFG.CAMERA_H)

    obj = OM.ObjectModel()
    obj.loadObjectCADModel(CFG.CAD_MODEL)
    obj.setIntrinsicMatrix(CFG.CAMERA_MATRIX)

    obj.determineSharpEdges(0.05)
    obj.generateSamplePoints(0.001, 0.001)

    current_index = 0
    try:
        with open(CFG.SAVE_PATH + "current_index_cache.txt") as index_cache:
            current_index = int(index_cache.read())
    except FileNotFoundError:
        pass
    while frame_captured:
        originImg = frame.copy()
        markers = detect_markers(frame)
        imagePoints = [None, None, None, None]
        objectPoints = [None, None, None, None]

        for marker in markers:
            marker.highlite_marker(frame)
            cv2.circle(frame, marker.center, 2, (0, 0, 255), -1)
            if marker.id == 1000:
                imagePoints[0] = [float(marker.center[0]), float(marker.center[1])]
                objectPoints[0] = tablePoints[0]
            elif marker.id == 2000:
                imagePoints[1] = [float(marker.center[0]), float(marker.center[1])]
                objectPoints[1] = tablePoints[1]
            elif marker.id == 3000:
                imagePoints[2] = [float(marker.center[0]), float(marker.center[1])]
                objectPoints[2] = tablePoints[2]
            elif marker.id == 4000:
                imagePoints[3] = [float(marker.center[0]), float(marker.center[1])]
                objectPoints[3] = tablePoints[3]

        imagePoints = list(filter(None, imagePoints))
        objectPoints = list(filter(None, objectPoints))

        if len(imagePoints) >= 4:

            cv2.putText(
                frame,
                "enough detection",
                (0, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            objectPoints = np.array(objectPoints)
            imagePoints = np.array(imagePoints)
            _, rvec, tvec = cv2.solvePnP(
                objectPoints, imagePoints, CFG.CAMERA_MATRIX, dist_coefs
            )
            rotMat, _ = cv2.Rodrigues(rvec)
            pose = np.identity(4)
            pose[:3, :3] = rotMat
            pose[0, 3] = tvec[0][0]
            pose[1, 3] = tvec[1][0]
            pose[2, 3] = tvec[2][0]

            pose = obj.setModelviewMatrix(pose)

            upperleft, lowerright = obj.findVisibleSamplePoint()
            upperleft = (int(upperleft[0]), int(upperleft[1]))
            lowerright = (int(lowerright[0]), int(lowerright[1]))

            pose = OM.symmetricRemove(pose)
            obj.setModelviewMatrix(pose)
            viewPoint, inplaneRotation, offsetFromCenter, depth = obj.getLabel()
            vpidx = OM.cal_idx(viewPoint)

            cv2.putText(
                frame,
                "view point " + str(vpidx),
                (0, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            for p in obj.sharp_2d_pts:
                frame = cv2.circle(frame, p, radius=0, color=(0, 0, 255), thickness=-1)

            cropImg = frame[upperleft[1] : lowerright[1], upperleft[0] : lowerright[0]]
            cropImg = cv2.resize(
                cropImg,
                (cropImg.shape[0] * 10, cropImg.shape[1] * 10),
                interpolation=cv2.INTER_AREA,
            )
            cv2.imshow("crop", cropImg)

            viewWindow = np.zeros(frame.shape, np.uint8)
            # display coordinate
            originPoint = obj.project3Dto2D((0.0, 0.0, 0.0), pose)
            xaxis = obj.project3Dto2D((0.05, 0.0, 0.0), pose)
            frame = cv2.line(frame, originPoint, xaxis, (255, 0, 0), 1)
            yaxis = obj.project3Dto2D((0.0, 0.05, 0.0), pose)
            frame = cv2.line(frame, originPoint, yaxis, (0, 255, 0), 1)
            zaxis = obj.project3Dto2D((0.0, 0.0, 0.05), pose)
            frame = cv2.line(frame, originPoint, zaxis, (0, 0, 255), 1)

            viewpoint = obj.getViewPoints(pose)
            viewpoint /= 10.0
            viewpoint2d = obj.project3Dto2D(viewpoint, pose)
            frame = cv2.circle(
                frame, viewpoint2d, radius=1, color=(255, 255, 255), thickness=-1
            )

            cv2.putText(
                frame,
                "Press c to save the image; press q to quit.",
                (0, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

            cv2.imshow("Test Frame", frame)

            ch = cv2.waitKey(1)
            if ch & 0xFF == ord("c"):  # collect data
                with open(
                    CFG.SAVE_PATH + "current_index_cache.txt", "w"
                ) as index_cache:
                    print(current_index, file=index_cache)
                cv2.imwrite(
                    CFG.SAVE_PATH + "{:06d}".format(current_index) + ".png", originImg
                )
                np.save(CFG.SAVE_PATH + "{:06d}".format(current_index) + ".npy", pose)
                current_index += 1

        else:
            cv2.putText(
                frame,
                "Press c to save the image; press q to quit.",
                (0, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                "No enough detection",
                (0, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            cv2.imshow("Test Frame", frame)

            ch = cv2.waitKey(1)

        if ch & 0xFF == ord("q"):
            break
        frame_captured, frame = capture.read()

    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()
