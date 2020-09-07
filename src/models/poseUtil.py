import numpy as np
import torch

import open3d as o3d
import torchgeometry as tgm
import kornia
import src.configuration as CFG
from scipy.spatial.transform import Rotation as R

# shape of inputs:
# inputPose (_,4,4)
# rot (_,3)
# trans(_,1,2)
# dist(_,1,1)
# imageSize 1
# rescaleValue(_)
def getPredictPose(initPose, rot, trans, dist, imagesize, rescaleValue):
    # convert the shift to right format
    trans = (trans - 0.5) * imagesize

    initPoseRot = torch.eye(4).repeat(trans.shape[0], 1, 1).cuda().float()
    initPoseRot[:, :3, :3] = initPose[:, :3, :3]

    # generate the rotation pose
    rot_pose = torch.bmm(tgm.angle_axis_to_rotation_matrix(rot), initPoseRot)
    rot_pose[:, :3, 3] = initPose[:, :3, 3]

    # apply the predicted trans and dist to the rot pose, so we can get
    # the predict pose
    dist_pose = rot_pose.clone()
    dist_pose[:, 2, 3] = rot_pose[:, 2, 3] / dist[:, 0, 0]

    horizontalR = torch.atan2(
        trans[:, :, 0].view(trans.shape[0]),
        torch.tensor(CFG.CAMERA_MATRIX[0, 0]).cuda() * rescaleValue,
    )

    verticalR = -torch.atan2(
        trans[:, :, 1].view(trans.shape[0]),
        torch.sqrt(
            trans[:, :, 0].view(trans.shape[0]) * trans[:, :, 0].view(trans.shape[0])
            + torch.tensor(CFG.CAMERA_MATRIX[1, 1] * CFG.CAMERA_MATRIX[1, 1]).cuda()
            * rescaleValue
            * rescaleValue
        )
        * torch.tensor(CFG.CAMERA_MATRIX[1, 1] / CFG.CAMERA_MATRIX[1, 1]).cuda(),
    )

    ch = torch.cos(horizontalR)
    sh = torch.sin(horizontalR)
    cb = torch.cos(verticalR)
    sb = torch.sin(verticalR)
    ca = torch.cos(torch.zeros(trans.shape[0]).cuda())  # z axiz
    sa = torch.sin(torch.zeros(trans.shape[0]).cuda())  # z axiz

    m00 = ch * ca
    m01 = sh * sb - ch * sa * cb
    m02 = ch * sa * sb + sh * cb
    m10 = sa
    m11 = ca * cb
    m12 = -ca * sb
    m20 = -sh * ca
    m21 = sh * sa * cb + ch * sb
    m22 = -sh * sa * sb + ch * cb

    rotation_matrix = torch.eye(4).repeat(trans.shape[0], 1, 1).cuda()
    m00 = m00.unsqueeze(0)
    m01 = m01.unsqueeze(0)
    m02 = m02.unsqueeze(0)
    m10 = m10.unsqueeze(0)
    m11 = m11.unsqueeze(0)
    m12 = m12.unsqueeze(0)
    m20 = m20.unsqueeze(0)
    m21 = m21.unsqueeze(0)
    m22 = m22.unsqueeze(0)

    m = torch.cat((m00, m01, m02, m10, m11, m12, m20, m21, m22), dim=0)
    m.transpose_(0, 1)
    rotation_matrix[..., :3, :3] = m.view(-1, 3, 3)

    # get predicted pose
    pred_pose = torch.bmm(rotation_matrix, dist_pose)

    return pred_pose


# shape of inputs:
# pred_pose (_,4,4)
# targetPose (_,4,4)
def getRotationError(pred_pose, targetPose):
    pred_rot = pred_pose[:, :3, :3]
    pred_rot_T = torch.transpose(pred_rot, 1, 2)
    target_rot = targetPose[:, :3, :3]
    rot_delta = torch.bmm(target_rot, pred_rot_T)
    rot_delta = rot_delta.data.cpu().numpy()
    rot_delta_angle_axis = np.array(
        [R.from_matrix(rot_mtx).as_rotvec() for rot_mtx in rot_delta]
    )
    rot_delta_angle = np.linalg.norm(rot_delta_angle_axis, axis=1)
    return rot_delta_angle


# shape of inputs:
# pred_pose (_,4,4)
# targetPose (_,4,4)
def ADD_error(pred_pose, targetPose):
    numberOfSamplePoints = 1000
    if pred_pose.shape[0] != targetPose.shape[0]:
        print("Error: the length of prediction and target are different!")
        return None
    # generate sample points on face of the object
    mesh = o3d.io.read_triangle_mesh(CFG.SAMPLE_FACE_MODEL)
    samplepoints = np.asarray(mesh.vertices)
    # get diameter
    maxdim = np.amax(samplepoints, axis=0)
    mindim = np.amin(samplepoints, axis=0)
    diameter = np.linalg.norm(maxdim - mindim)
    if samplepoints.shape[0] > numberOfSamplePoints:
        samplepointind = np.random.choice(
            samplepoints.shape[0], numberOfSamplePoints, replace=False
        )
        samplepoints = torch.from_numpy(samplepoints[samplepointind]).cuda().float()
        samplepoints = samplepoints.repeat(pred_pose.shape[0], 1, 1)
    else:
        print("points on face is not enough!")
        return None

    predict_points = tgm.transform_points(pred_pose, samplepoints)
    target_points = tgm.transform_points(targetPose, samplepoints)
    distanceBetweenVec3d = predict_points - target_points
    dist3d = torch.norm(distanceBetweenVec3d, p=1, dim=2)
    result = dist3d < (diameter * 0.1)
    result = torch.sum(result, dim=1).float() / numberOfSamplePoints
    return result
