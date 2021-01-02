#!/home/cogrob-wrc/miniconda3/envs/pose-estimation/bin/python3
from src.models.detect import detect, vs_detect
import roslibpy
import base64
import numpy as np
import cv2


def get_pose(object_id, img, estimated_depth):
    result, confidence, status = detect(object_id, img, estimated_depth)
    print(result)
    if result is not None:
        result = result.reshape(16,)
    return result, confidence, status


def handler(request, response):
    print(f"Object ID is: {request['object_id']}")

    base64_bytes = request["image"]["data"].encode("ascii")
    image_bytes = base64.b64decode(base64_bytes)

    nparr = np.fromstring(image_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    out, confidence, status = get_pose(
        request["object_id"], img_np, float(request["estimated_depth"])
    )
    print(out)
    if out is None:
        response["row_major_hmatrix"] = ""
    else:
        response["row_major_hmatrix"] = f"{out.tolist()}"
    response["confidence"] = confidence
    response["status"] = status

    return True


def get_vs_pose(object_id, img):
    result, camera_horizontalR, camera_verticalR, status = vs_detect(object_id, img)
    if result is not None:
        result = result.reshape(16,)
    return result, camera_horizontalR, camera_verticalR, status


def vs_handler(request, response):
    if request["object_id"] == "":
        print("receive test request!")
        return False
    print(f"Object ID is: {request['object_id']}")

    base64_bytes = request["image"]["data"].encode("ascii")
    image_bytes = base64.b64decode(base64_bytes)

    nparr = np.fromstring(image_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    out, camera_horizontalR, camera_verticalR, status = get_vs_pose(
        request["object_id"], img_np
    )
    print(out)
    if out is not None:
        response["row_major_hmatrix"] = f"{out.tolist()}"
    else:
        response["row_major_hmatrix"] = ""
    response["horizontal_rotate"] = camera_horizontalR
    response["vertical_rotate"] = camera_verticalR
    response["status"] = status

    return True


ROS_PC_IP = "192.168.1.6"


def main():
    client = roslibpy.Ros(host=ROS_PC_IP, port=9090)

    right_service = roslibpy.Service(
        client, "right_ur_robot/estimate_pose", "wrc_msgs/EstimatePose"
    )
    right_service.advertise(handler)

    left_service = roslibpy.Service(
        client, "left_ur_robot/estimate_pose", "wrc_msgs/EstimatePose"
    )
    left_service.advertise(handler)
    print("Left service advertised.")

    right_visual_servo_service = roslibpy.Service(
        client, "right_ur_robot/visual_servo_pose", "wrc_msgs/VisualServo"
    )
    right_visual_servo_service.advertise(vs_handler)

    left_visual_servo_service = roslibpy.Service(
        client, "left_ur_robot/visual_servo_pose", "wrc_msgs/VisualServo"
    )
    left_visual_servo_service.advertise(vs_handler)

    client.run_forever()
    client.terminate()


if __name__ == "__main__":
    main()
