#!/home/cogrob-wrc/miniconda3/envs/pose-estimation/bin/python3
from src.models.detect import detect
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

    out, confidence = get_pose(request["object_id"], img_np, float(request["estimated_depth"]))
    print(out)
    response["row_major_hmatrix"] = f"{out.tolist()}"
    response["confidence"] = confidence

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

    client.run_forever()
    client.terminate()


if __name__ == "__main__":
    main()
