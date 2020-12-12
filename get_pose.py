#!/home/cogrob-wrc/miniconda3/envs/pose-estimation/bin/python3
from src.models.detect import detect, vs_detect
import roslibpy
import base64
import numpy as np
import cv2


def get_pose(object_id, img, estimated_depth):
    result, confidence = detect(object_id, img, estimated_depth)
    print(result)
    result = result.reshape(16,)
    return result, confidence


def handler(request, response):
    print(f"Object ID is: {request['object_id']}")

    base64_bytes = request["image"]["data"].encode("ascii")
    image_bytes = base64.b64decode(base64_bytes)

    nparr = np.fromstring(image_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    out, confidence = get_pose(
        request["object_id"], img_np, float(request["estimated_depth"])
    )
    print(out)
    response["row_major_hmatrix"] = f"{out.tolist()}"
    response["confidence"] = confidence

    return True


def get_vs_pose(object_id, img):
    result, status = vs_detect(object_id, img)
    result = result.reshape(16,)
    return result, status


def vs_handler(request, response):
    if request["object_id"] == "":
        print("receive test request!")
        return False
    print(f"Object ID is: {request['object_id']}")

    base64_bytes = request["image"]["data"].encode("ascii")
    image_bytes = base64.b64decode(base64_bytes)

    nparr = np.fromstring(image_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    out, status = get_vs_pose(request["object_id"], img_np)
    print(out)
    response["row_major_hmatrix"] = f"{out.tolist()}"
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
