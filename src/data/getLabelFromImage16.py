from __future__ import print_function
import numpy as np
import cv2
from ar_markers import detect_markers
import src.common.object_model as OM
import src.configuration as CFG
import os
from scipy.spatial.transform import Rotation as R

# ignore warming
np.seterr(divide="ignore", invalid="ignore")

# you need to update the table points for your case###
tablex = 0.04
tabley = 0.0
tablez = 0.05
tablePoints_base = [
    [0.019565, 0.0, -0.019565],
    [0.019565, 0.0, -0.006521],
    [0.019565, 0.0, 0.006521],
    [0.019565, 0.0, 0.019565],
    [0.006521, 0.0, -0.019565],
    [0.006521, 0.0, -0.006521],
    [0.006521, 0.0, 0.006521],
    [0.006521, 0.0, 0.019565],
    [-0.006521, 0.0, -0.019565],
    [-0.006521, 0.0, -0.006521],
    [-0.006521, 0.0, 0.006521],
    [-0.006521, 0.0, 0.019565],
    [-0.019565, 0.0, -0.019565],
    [-0.019565, 0.0, -0.006521],
    [-0.019565, 0.0, 0.006521],
    [-0.019565, 0.0, 0.019565],
]
#######################################################

dist_coefs = np.zeros((4, 1))

if __name__ == "__main__":
    # ensure the directory which store images and poses exists
    if not os.path.isdir(CFG.IMAGE_SAVE_PATH):
        print("please make a directory:")
        print("mkdir", CFG.IMAGE_SAVE_PATH)
        exit()

    # init camera
    capture = cv2.VideoCapture(CFG.CAMERA_ID)
    capture.set(3, CFG.CAMERA_W)
    capture.set(4, CFG.CAMERA_H)

    # init object model
    OM.setup(CFG.CAMERA_W, CFG.CAMERA_H)

    OM.setProjectMatrixWithIntr(CFG.CAMERA_MATRIX, CFG.CAMERA_W, CFG.CAMERA_H)

    obj = OM.ObjectModel()
    obj.loadObjectCADModel(CFG.CAD_MODEL)
    obj.setIntrinsicMatrix(CFG.CAMERA_MATRIX)

    obj.determineSharpEdges(0.8)
    obj.generateSamplePoints(0.0001)

    current_index = 0
    # read the index if it exists
    try:
        with open(CFG.IMAGE_SAVE_PATH + "current_index_cache.txt") as index_cache:
            current_index = int(index_cache.read())
    except FileNotFoundError:
        pass

    gotPose = False
    pose = None
    usingMarker = False

    while True:
        transform = np.identity(4)
        transform[:3, :3] = (R.from_euler("z", 0)).as_matrix()
        transform[0][3] = tablex
        transform[1][3] = tabley
        transform[2][3] = tablez
        points = np.array(tablePoints_base)
        points_h = np.append(points, np.ones((points.shape[0], 1)), 1).T
        tablePoints = np.delete(transform.dot(points_h).T, -1, axis=1).tolist()

        if capture.isOpened():  # try to get the first frame
            _, frame = capture.read()
        else:
            print("no image coming in!!!")
            break

        originImg = frame.copy()
        if usingMarker and gotPose:
            cv2.putText(
                frame,
                "Keep current pose!",
                (0, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        elif not usingMarker:
            # using the marker in the image to infer the pose of the object
            markers = detect_markers(frame)
            imagePoints = [
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ]
            objectPoints = [
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ]

            for marker in markers:
                marker.highlite_marker(frame, linewidth=1, text_thickness=1)
                cv2.circle(frame, marker.center, 2, (0, 0, 255), -1)
                for b in range(16):
                    if marker.id == b + 1:
                        imagePoints[b] = [
                            float(marker.center[0]),
                            float(marker.center[1]),
                        ]
                        objectPoints[b] = tablePoints[b]
                        break

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
                _, rvec, tvec, _ = cv2.solvePnPRansac(
                    objectPoints,
                    imagePoints,
                    CFG.CAMERA_MATRIX,
                    dist_coefs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                rotMat, _ = cv2.Rodrigues(rvec)
                pose = np.identity(4)
                pose[:3, :3] = rotMat
                pose[0, 3] = tvec[0][0]
                pose[1, 3] = tvec[1][0]
                pose[2, 3] = tvec[2][0]
                gotPose = True
            else:
                cv2.putText(
                    frame,
                    "No enough detection",
                    (CFG.CAMERA_W - 400, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                gotPose = False

        if gotPose:
            pose = obj.setModelviewMatrix(pose)
            obj.findVisibleSamplePoint()
            bx, by, bw, bh = cv2.boundingRect(obj.getVisibleArea())
            if bh == 0 or bw == 0:
                continue

            boundingsize = max(bw, bh) * CFG.EXPAND_SIZE

            # get center point from pose
            centerPoint = obj.project3Dto2D((0, 0, 0), pose)

            ex = int(centerPoint[0] - boundingsize / 2)
            ey = int(centerPoint[1] - boundingsize / 2)
            ew = int(boundingsize)
            eh = int(boundingsize)

            upperleft = (ex, ey)
            lowerright = (ex + ew, ey + eh)

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
                p = (int(p[0]), int(p[1]))
                frame = cv2.circle(frame, p, radius=1, color=(0, 0, 255), thickness=-1)

            # if you think the object is too small, you can uncomment following code to zoom the object in another window
            # #  adjust the bounding box
            # crop_upperleft, crop_lowerright = OM.get_centered_crop(
            #     upperleft, lowerright
            # )
            # cropImg = np.zeros(
            #     (
            #         crop_lowerright[1] - crop_upperleft[1],
            #         crop_lowerright[0] - crop_upperleft[0],
            #         3,
            #     ),
            #     np.uint8,
            # )
            # upperleft_crop_inner = [
            #     max(0, crop_upperleft[0]),
            #     max(0, crop_upperleft[1]),
            # ]
            # lowerright_crop_inner = [
            #     min(frame.shape[1], crop_lowerright[0]),
            #     min(frame.shape[0], crop_lowerright[1]),
            # ]
            # cropImg[
            #     upperleft_crop_inner[1]
            #     - crop_upperleft[1] : lowerright_crop_inner[1]
            #     - crop_upperleft[1],
            #     upperleft_crop_inner[0]
            #     - crop_upperleft[0] : lowerright_crop_inner[0]
            #     - crop_upperleft[0],
            # ] = frame[
            #     int(upperleft_crop_inner[1]) : int(lowerright_crop_inner[1]),
            #     int(upperleft_crop_inner[0]) : int(lowerright_crop_inner[0]),
            # ]

            # cropImg = cv2.resize(
            #     cropImg,
            #     (cropImg.shape[1], cropImg.shape[0]),
            #     interpolation=cv2.INTER_AREA,
            # )
            # cv2.imshow("crop", cropImg)

            # display coordinate
            originPoint = obj.project3Dto2D((0.0, 0.0, 0.0), pose)
            originPoint = tuple(map(int, originPoint))
            xaxis = obj.project3Dto2D((0.05, 0.0, 0.0), pose)
            xaxis = tuple(map(int, xaxis))
            frame = cv2.line(frame, originPoint, xaxis, (255, 0, 0), 1)
            yaxis = obj.project3Dto2D((0.0, 0.05, 0.0), pose)
            yaxis = tuple(map(int, yaxis))
            frame = cv2.line(frame, originPoint, yaxis, (0, 255, 0), 1)
            zaxis = obj.project3Dto2D((0.0, 0.0, 0.05), pose)
            zaxis = tuple(map(int, zaxis))
            frame = cv2.line(frame, originPoint, zaxis, (0, 0, 255), 1)

            viewpoint = obj.getViewPoints(pose)
            viewpoint /= 10.0
            viewpoint = tuple(viewpoint)
            viewpoint2d = obj.project3Dto2D(viewpoint, pose)
            viewpoint2d = tuple(map(int, viewpoint2d))
            frame = cv2.circle(
                frame, viewpoint2d, radius=1, color=(255, 255, 255), thickness=-1
            )
        else:
            cv2.putText(
                frame,
                "There is no pose currently!!",
                (0, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        cv2.putText(
            frame,
            "Press c to save the image; press q to quit.",
            (0, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
        )
        cv2.imshow("Test Frame", frame)
        ch = cv2.waitKey(1)
        if ch & 0xFF == ord("c"):  # collect data
            if gotPose:
                with open(
                    CFG.IMAGE_SAVE_PATH + "current_index_cache.txt", "w"
                ) as index_cache:
                    print(current_index, file=index_cache)
                cv2.imwrite(
                    CFG.IMAGE_SAVE_PATH + "{:06d}".format(current_index) + ".png",
                    originImg,
                )
                np.save(
                    CFG.IMAGE_SAVE_PATH + "{:06d}".format(current_index) + ".npy", pose
                )
                current_index += 1
            else:
                print("where is no pose!")
        elif ch & 0xFF == ord("k"):
            usingMarker = not usingMarker
        elif ch & 0xFF == ord("q"):
            break
        elif ch & 0xFF == ord("m"):
            tabley -= 0.001
        elif ch & 0xFF == ord("n"):
            tabley += 0.001

    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()
