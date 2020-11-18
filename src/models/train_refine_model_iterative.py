from src.common.DataLoader import Iterative_refine_data
from pathlib import Path
from models.model import DeepIM, FlowNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import random
import numpy as np
from src.common.iou import iou
from poseUtil import getPredictPose, getRotationError, ADD_error, ADDS_error
from torch.multiprocessing import Pool, Value, cpu_count, Array
import torchgeometry as tgm
import kornia
import src.common.object_model as OM
import src.configuration as CFG
import cv2
import open3d as o3d
from ctypes import c_bool

# importing shutil module
import shutil

# allow multiple processes to access the file system
torch.multiprocessing.set_sharing_strategy("file_system")

# initial global varible for processes
counter = Value("i", 0)
output_counter = Value("i", 0)
testTrigger = Value(c_bool, False)


# refine parameters
batch_size = 64
epochs = 120
lr = 4e-5
momentum = 0.9
w_decay = 0.1
seglambda = 0.5
flowlambda = 5.0

# file addresses
train_dir = CFG.REFINE_ITERATIVE_DATA_PATH
val_dir = CFG.REFINE_ITERATIVE_DATA_PATH
processed_dir = "temp/"
pool_dir = "pred_temp/"

# initiate the net
mymodel = DeepIM().cuda()
mymodel = nn.DataParallel(mymodel)
# mymodel.module.flownet.load_state_dict(
#     torch.load(CFG.BEST_MODEL_FLOWNET).module.state_dict()
# )
# mymodel.module.flownet.eval()

mymodel = torch.load(CFG.BEST_MODEL_ITERATIVE_REFINE)

seg_criterion = nn.CrossEntropyLoss(reduce=False)

optimizer = optim.Adam(
    mymodel.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-08, weight_decay=w_decay
)
lmbda = lambda epoch: 0.5
scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

torch.autograd.set_detect_anomaly(True)

# get diameter of model
samplepoints = np.asarray(o3d.io.read_triangle_mesh(CFG.CAD_MODEL).vertices)
diameter = np.linalg.norm(np.amax(samplepoints, axis=0) - np.amin(samplepoints, axis=0))

# object init.
def init(sampleValue):
    # load the object mesh
    OM.setup(CFG.CAMERA_W, CFG.CAMERA_H)
    OM.setProjectMatrixWithIntr(CFG.CAMERA_MATRIX, CFG.CAMERA_W, CFG.CAMERA_H)
    obj = OM.ObjectModel()
    obj.setIntrinsicMatrix(CFG.CAMERA_MATRIX)
    obj.loadObjectCADModel(CFG.CAD_MODEL)

    obj.determineSharpEdges(0.8)
    obj.generateSamplePoints(0.00001, sampleValue)
    return obj


# preprocess the init pose, render the input for module.
def process_data(args):
    global counter
    global output_counter
    global testTrigger

    obj = init(0.0001)

    # parse input
    (id, datalist) = args
    (img_names, initpose_names, targetpose_names, mask_names) = list(zip(*datalist))

    while True:
        if testTrigger.value == True:
            break
        try:
            with counter.get_lock():
                current_index = counter.value
                counter.value += 1

            if current_index >= len(datalist):
                return

            # update the progress bar
            progress = int(50.0 * current_index / len(datalist))
            rest_progress = 50 - progress
            print(
                "Progress: ["
                + "=" * progress
                + " " * rest_progress
                + "]"
                + str(100.0 * current_index / len(datalist))
                + "%",
                end="\r",
                flush=True,
            )

            # read image and pose
            img = cv2.imread(img_names[current_index])
            initpose = np.load(initpose_names[current_index])
            targetpose = np.load(targetpose_names[current_index])
            target_mask = cv2.imread(mask_names[current_index])

            # get current pose if it is moved to the center
            horizontalR, verticalR = obj.getCenterAngle(initpose)

            # set pose on object
            obj.setModelviewMatrix(initpose)

            # generate edge of on the object
            obj.findVisibleSamplePoint()

            # generate preprocessed data
            # inital pose mask(it may not return any mask because
            # the init pose is too close to the camera)
            init_mask = obj.getVisibleArea()

            # if the init pose is too close, then continue
            if cv2.countNonZero(init_mask) == 0:
                continue

            # get edge img
            edge = obj.getEdge(img.shape[0], img.shape[1])

            # find the crop size
            [_, _, w, h] = cv2.boundingRect(init_mask)

            boundingsize = max(w, h) * CFG.EXPAND_SIZE

            # generate the optical flow from intial pose to target pose
            flowImg = obj.getOptFlowWithPoses(boundingsize, boundingsize, targetpose)

            # get center point from pose
            centerPoint = obj.project3Dto2D((0, 0, 0), initpose)

            ex = int(centerPoint[0] - boundingsize / 2)
            ey = int(centerPoint[1] - boundingsize / 2)
            ew = int(boundingsize)
            eh = int(boundingsize)

            crop_img = np.zeros((eh, ew, 3), np.uint8,)
            crop_init_mask = np.zeros((eh, ew), np.uint8)
            crop_edge = np.zeros((eh, ew), np.uint8)
            crop_label_mask = np.zeros((eh, ew, 3), np.uint8)
            crop_flowImg = np.zeros((eh, ew, 3), np.uint8)

            upperleft_crop_inner = [max(0, ex), max(0, ey)]
            lowerright_crop_inner = [
                min(img.shape[1], ex + ew),
                min(img.shape[0], ey + eh),
            ]

            # cropped image with initial pose as center
            crop_img[
                upperleft_crop_inner[1] - ey : lowerright_crop_inner[1] - ey,
                upperleft_crop_inner[0] - ex : lowerright_crop_inner[0] - ex,
            ] = img[
                int(upperleft_crop_inner[1]) : int(lowerright_crop_inner[1]),
                int(upperleft_crop_inner[0]) : int(lowerright_crop_inner[0]),
            ]

            crop_init_mask[
                upperleft_crop_inner[1] - ey : lowerright_crop_inner[1] - ey,
                upperleft_crop_inner[0] - ex : lowerright_crop_inner[0] - ex,
            ] = init_mask[
                int(upperleft_crop_inner[1]) : int(lowerright_crop_inner[1]),
                int(upperleft_crop_inner[0]) : int(lowerright_crop_inner[0]),
            ]

            crop_edge[
                upperleft_crop_inner[1] - ey : lowerright_crop_inner[1] - ey,
                upperleft_crop_inner[0] - ex : lowerright_crop_inner[0] - ex,
            ] = edge[
                int(upperleft_crop_inner[1]) : int(lowerright_crop_inner[1]),
                int(upperleft_crop_inner[0]) : int(lowerright_crop_inner[0]),
            ]

            crop_label_mask[
                upperleft_crop_inner[1] - ey : lowerright_crop_inner[1] - ey,
                upperleft_crop_inner[0] - ex : lowerright_crop_inner[0] - ex,
            ] = target_mask[
                int(upperleft_crop_inner[1]) : int(lowerright_crop_inner[1]),
                int(upperleft_crop_inner[0]) : int(lowerright_crop_inner[0]),
            ]

            crop_flowImg[
                upperleft_crop_inner[1] - ey : lowerright_crop_inner[1] - ey,
                upperleft_crop_inner[0] - ex : lowerright_crop_inner[0] - ex,
            ] = flowImg[
                int(upperleft_crop_inner[1]) : int(lowerright_crop_inner[1]),
                int(upperleft_crop_inner[0]) : int(lowerright_crop_inner[0]),
            ]

            # apply rotation to initial pose
            init_pose_at_center = obj.rotatePoseWithAngle(
                initpose, horizontalR, verticalR
            )

            # apply same rotation to target pose
            target_pose_at_center = obj.rotatePoseWithAngle(
                targetpose, horizontalR, verticalR
            )

            with output_counter.get_lock():
                current_output_index = output_counter.value
                output_counter.value += 1

            cv2.imwrite(
                processed_dir
                + "{:06d}".format(current_output_index)
                + "img_original.png",
                img,
            )
            cv2.imwrite(
                processed_dir
                + "{:06d}".format(current_output_index)
                + "labelmask_original.png",
                target_mask,
            )

            np.save(
                processed_dir
                + "{:06d}".format(current_output_index)
                + "initPose_original.npy",
                initpose,
            )

            np.save(
                processed_dir
                + "{:06d}".format(current_output_index)
                + "targetPose_original.npy",
                targetpose,
            )

            cv2.imwrite(
                processed_dir + "{:06d}".format(current_output_index) + "img.png",
                crop_img,
            )
            cv2.imwrite(
                processed_dir + "{:06d}".format(current_output_index) + "mask.png",
                crop_init_mask,
            )
            cv2.imwrite(
                processed_dir + "{:06d}".format(current_output_index) + "edge.png",
                crop_edge,
            )

            cv2.imwrite(
                processed_dir + "{:06d}".format(current_output_index) + "labelmask.png",
                crop_label_mask,
            )
            cv2.imwrite(
                processed_dir + "{:06d}".format(current_output_index) + "flow.png",
                crop_flowImg,
            )
            np.save(
                processed_dir + "{:06d}".format(current_output_index) + "initPose.npy",
                init_pose_at_center,
            )
            np.save(
                processed_dir
                + "{:06d}".format(current_output_index)
                + "targetPose.npy",
                target_pose_at_center,
            )

        except Exception as e:
            print("fail in pre-processing")
            print(str(e))
            testTrigger.value = True

    OM.exit()


def val(sample_points, val_loader, val_dataset):
    mymodel.eval()
    avg_loss = []
    avg_rot_error = []
    avg_trans_error = []
    avg_add_match_rate = []
    avg_adds_match_rate = []
    avg_iou = []

    for data in val_loader:
        (
            idx,
            input_img,
            edge_img,
            mask_img,
            initPose,
            targetPose,
            labelmask_img,
            flow_img,
            rescaleValue,
        ) = data

        object_sample_points = sample_points.repeat(input_img.shape[0], 1, 1)

        input_img = input_img.cuda().float() / 255.0
        edge_img = edge_img.cuda().float() / 255.0
        mask_img = mask_img.cuda().float() / 255.0
        object_sample_points = object_sample_points.cuda().float()
        initPose = initPose.cuda().float()
        targetPose = targetPose.cuda().float()
        rescaleValue = rescaleValue.cuda().float()
        labelflow = flow_img[:, :2, :, :].cuda().float()

        labelmask_img = Variable(labelmask_img).cuda().long()
        labelmask_img = labelmask_img.squeeze(1)

        flow_inputData = torch.cat((mask_img, edge_img, input_img), 1,)

        flow_input = Variable(flow_inputData)

        with torch.no_grad():
            rot, trans, dist, opticalFlow, segmentMask = mymodel(flow_input)

        opticalFlow = torch.sigmoid(opticalFlow)

        segmentMask = segmentMask.squeeze(1)

        segloss = seg_criterion(segmentMask, labelmask_img)
        segloss = torch.mean(segloss.view(segloss.shape[0], -1), dim=1)
        segloss = segloss.sum() * seglambda

        maskarea = torch.sum(mask_img.view(mask_img.shape[0], -1), dim=1)

        flowloss = torch.norm(opticalFlow - labelflow, p=1, dim=1)
        flowloss = flowloss.unsqueeze(1)
        flowloss = flowloss * mask_img
        flowloss = torch.sum(flowloss.view(flowloss.shape[0], -1), dim=1)
        flowloss = flowloss / maskarea

        flowloss = flowloss.sum() * flowlambda

        trans = trans.unsqueeze(1)
        dist = dist.unsqueeze(1)

        # update the camera matrix because the input image is resize
        camera_matrix_original = (
            torch.tensor(
                [
                    [
                        [CFG.CAMERA_MATRIX[0, 0], 0.0, float(input_img.shape[-2]) / 2,],
                        [0.0, CFG.CAMERA_MATRIX[1, 1], float(input_img.shape[-2]) / 2,],
                        [0.0, 0.0, 1.0],
                    ]
                ]
            )
            .repeat(trans.shape[0], 1, 1)
            .cuda()
        )
        camera_matrix = camera_matrix_original.clone()
        camera_matrix[:, 0, 0] = camera_matrix_original[:, 0, 0] * rescaleValue
        camera_matrix[:, 1, 1] = camera_matrix_original[:, 1, 1] * rescaleValue
        camera_matrix.unsqueeze_(1)

        pred_pose = getPredictPose(
            initPose, rot, trans, dist, input_img.shape[-2], rescaleValue
        )

        # predict 3d pts
        predict3dpts = tgm.transform_points(pred_pose, object_sample_points)

        # rotation error
        rotError = getRotationError(pred_pose, targetPose)

        # translation error
        transError = torch.norm(pred_pose[:, :3, 3] - targetPose[:, :3, 3], p=2, dim=1)

        # ADD rate
        addMatchRate = ADD_error(pred_pose, targetPose)
        # ADDS rate
        addsMatchRate = ADDS_error(pred_pose, targetPose)

        origin_pt = torch.tensor([0, 0, 0]).view(1, 1, 3).cuda().float()
        origin_pt = origin_pt.repeat(
            object_sample_points.shape[0], object_sample_points.shape[1], 1
        )
        origin_pt = tgm.transform_points(targetPose, origin_pt)

        # get distance between each point pairs in 3d
        label3dPt = tgm.transform_points(targetPose, object_sample_points)
        predictVec = predict3dpts - origin_pt
        targetVec = label3dPt - origin_pt
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        cosineSim = cos(predictVec, targetVec)
        cosloss = -torch.mean(cosineSim, 1).sum()

        # loss = dist3dloss + flowloss + segloss
        loss = cosloss + flowloss + segloss

        iou_value = iou(labelmask_img, segmentMask, 1)

        avg_loss.append(loss.data.cpu().numpy().sum())
        avg_rot_error.append(rotError.sum())
        avg_trans_error.append(transError.data.cpu().numpy().sum())
        avg_add_match_rate.append(addMatchRate.sum())
        avg_adds_match_rate.append(addsMatchRate.sum())
        avg_iou.append(iou_value.data.cpu().numpy())
    tem = sum(avg_loss) / len(val_dataset)
    tem_rot_error = sum(avg_rot_error) / len(val_dataset)
    tem_trans_error = sum(avg_trans_error) / len(val_dataset)
    term_add_rate = sum(avg_add_match_rate).float() / len(val_dataset)
    term_adds_rate = sum(avg_adds_match_rate).float() / len(val_dataset)
    ioutem = sum(avg_iou) / len(val_dataset)

    print(
        "val loss {}, rot_error {}, trans_error {} iou {} add rate {}% adds rate {}%".format(
            tem,
            tem_rot_error,
            tem_trans_error,
            ioutem,
            term_add_rate * 100,
            term_adds_rate * 100,
        )
    )
    return tem


def train(sample_points, train_loader, train_dataset):
    mymodel.train()
    avg_loss = []
    for data in train_loader:
        (
            idx,
            input_img,
            edge_img,
            mask_img,
            initPose,
            targetPose,
            labelmask_img,
            flow_img,
            rescaleValue,
        ) = data

        object_sample_points = sample_points.repeat(input_img.shape[0], 1, 1)

        # load data to cuda
        input_img = input_img.cuda().float() / 255.0
        edge_img = edge_img.cuda().float() / 255.0
        mask_img = mask_img.cuda().float() / 255.0
        initPose = initPose.cuda().float()
        object_sample_points = object_sample_points.cuda().float()
        targetPose = targetPose.cuda().float()
        rescaleValue = rescaleValue.cuda().float()
        labelflow = flow_img[:, :2, :, :].cuda().float()

        labelmask_img = Variable(labelmask_img).cuda().long()
        labelmask_img = labelmask_img.squeeze(1)

        flow_inputData = torch.cat((mask_img, edge_img, input_img), 1,)
        flow_input = Variable(flow_inputData)

        optimizer.zero_grad()

        # predict the rotation, translation, and depth in image view
        rot, trans, dist, opticalFlow, segmentMask = mymodel(flow_input)

        opticalFlow = torch.sigmoid(opticalFlow)

        segmentMask = segmentMask.squeeze(1)

        segloss = seg_criterion(segmentMask, labelmask_img)
        segloss = torch.mean(segloss.view(segloss.shape[0], -1), dim=1)
        segloss = segloss.sum() * seglambda

        maskarea = torch.sum(mask_img.view(mask_img.shape[0], -1), dim=1)

        flowloss = torch.norm(opticalFlow - labelflow, p=1, dim=1)
        flowloss = flowloss.unsqueeze(1)
        flowloss = flowloss * mask_img
        flowloss = torch.sum(flowloss.view(flowloss.shape[0], -1), dim=1)
        flowloss = flowloss / maskarea

        flowloss = flowloss.sum() * flowlambda

        trans = trans.unsqueeze(1)
        dist = dist.unsqueeze(1)

        # update the camera matrix because the input image is resize
        camera_matrix_original = (
            torch.tensor(
                [
                    [
                        [CFG.CAMERA_MATRIX[0, 0], 0.0, float(input_img.shape[-2]) / 2,],
                        [0.0, CFG.CAMERA_MATRIX[1, 1], float(input_img.shape[-2]) / 2,],
                        [0.0, 0.0, 1.0],
                    ]
                ]
            )
            .repeat(trans.shape[0], 1, 1)
            .cuda()
        )
        camera_matrix = camera_matrix_original.clone()
        camera_matrix[:, 0, 0] = camera_matrix_original[:, 0, 0] * rescaleValue
        camera_matrix[:, 1, 1] = camera_matrix_original[:, 1, 1] * rescaleValue
        camera_matrix.unsqueeze_(1)

        pred_pose = getPredictPose(
            initPose, rot, trans, dist, input_img.shape[-2], rescaleValue
        )

        # predict 3d pts
        predict3dpts = tgm.transform_points(pred_pose, object_sample_points)

        origin_pt = torch.tensor([0, 0, 0]).view(1, 1, 3).cuda().float()
        origin_pt = origin_pt.repeat(
            object_sample_points.shape[0], object_sample_points.shape[1], 1
        )
        origin_pt = tgm.transform_points(targetPose, origin_pt)

        # get distance between each point pairs in 3d
        label3dPt = tgm.transform_points(targetPose, object_sample_points)
        predictVec = predict3dpts - origin_pt
        targetVec = label3dPt - origin_pt
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        cosineSim = cos(predictVec, targetVec)
        cosloss = -torch.mean(cosineSim, 1).sum()

        loss = cosloss + flowloss + segloss

        loss.backward()
        optimizer.step()
        avg_loss.append(loss.data.cpu().numpy().sum())

    tem = sum(avg_loss) / len(train_dataset)
    return tem


# generate the init pose for next round
def generateData(obj, sample_points):
    mymodel = torch.load(CFG.BEST_MODEL_ITERATIVE_REFINE)
    mymodel.eval()

    val_dataset = Iterative_refine_data(data_path=processed_dir, isTrain=False)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )
    for data in val_loader:
        (
            idx,
            input_img,
            edge_img,
            mask_img,
            initPose,
            targetPose,
            labelmask_img,
            flow_img,
            rescaleValue,
        ) = data

        object_sample_points = sample_points.repeat(input_img.shape[0], 1, 1)

        input_img = input_img.cuda().float() / 255.0
        edge_img = edge_img.cuda().float() / 255.0
        mask_img = mask_img.cuda().float() / 255.0
        object_sample_points = object_sample_points.cuda().float()
        initPose = initPose.cuda().float()
        targetPose = targetPose.cuda().float()
        rescaleValue = rescaleValue.cuda().float()
        labelflow = flow_img[:, :2, :, :].cuda().float()

        labelmask_img = Variable(labelmask_img).cuda().long()
        labelmask_img = labelmask_img.squeeze(1)

        flow_inputData = torch.cat((mask_img, edge_img, input_img), 1,)

        flow_input = Variable(flow_inputData)

        with torch.no_grad():
            rot, trans, dist, opticalFlow, segmentMask = mymodel(flow_input)

        opticalFlow = torch.sigmoid(opticalFlow)

        segmentMask = segmentMask.squeeze(1)

        segloss = seg_criterion(segmentMask, labelmask_img)
        segloss = torch.mean(segloss.view(segloss.shape[0], -1), dim=1)
        segloss = segloss.sum() * seglambda

        maskarea = torch.sum(mask_img.view(mask_img.shape[0], -1), dim=1)

        flowloss = torch.norm(opticalFlow - labelflow, p=1, dim=1)
        flowloss = flowloss.unsqueeze(1)
        flowloss = flowloss * mask_img
        flowloss = torch.sum(flowloss.view(flowloss.shape[0], -1), dim=1)
        flowloss = flowloss / maskarea

        flowloss = flowloss.sum() * flowlambda

        trans = trans.unsqueeze(1)
        dist = dist.unsqueeze(1)

        # update the camera matrix because the input image is resize
        camera_matrix_original = (
            torch.tensor(
                [
                    [
                        [CFG.CAMERA_MATRIX[0, 0], 0.0, float(input_img.shape[-2]) / 2,],
                        [0.0, CFG.CAMERA_MATRIX[1, 1], float(input_img.shape[-2]) / 2,],
                        [0.0, 0.0, 1.0],
                    ]
                ]
            )
            .repeat(trans.shape[0], 1, 1)
            .cuda()
        )
        camera_matrix = camera_matrix_original.clone()
        camera_matrix[:, 0, 0] = camera_matrix_original[:, 0, 0] * rescaleValue
        camera_matrix[:, 1, 1] = camera_matrix_original[:, 1, 1] * rescaleValue
        camera_matrix.unsqueeze_(1)

        pred_pose = (
            getPredictPose(
                initPose, rot, trans, dist, input_img.shape[-2], rescaleValue
            )
            .cpu()
            .numpy()
        )

        for predindex in range(pred_pose.shape[0]):

            # read init pose
            savedIndex = idx[predindex].cpu().numpy()
            init_original_pose = np.load(
                processed_dir + "{:06d}".format(savedIndex) + "initPose_original.npy"
            )

            original_img = cv2.imread(
                processed_dir + "{:06d}".format(savedIndex) + "img_original.png"
            )

            # copy original image
            shutil.copyfile(
                processed_dir + "{:06d}".format(savedIndex) + "img_original.png",
                pool_dir + "{:06d}".format(savedIndex) + "img.png",
            )

            horizontalR_ori, verticalR_ori = obj.getCenterAngle(init_original_pose)

            global_pred_pose = obj.rotatePoseWithAngle(
                pred_pose[predindex], -horizontalR_ori, -verticalR_ori
            )

            # randomly resample the pose
            if bool(random.getrandbits(1)):
                global_pred_pose = obj.resamplePose(
                    global_pred_pose, diameter * 0.08, diameter * 0.15, 0.27
                )

            obj.setModelviewMatrix(global_pred_pose)
            obj.findVisibleSamplePoint()
            for p in obj.sharp_2d_pts:
                p = (int(p[0]), int(p[1]))
                original_img = cv2.circle(
                    original_img, p, radius=1, color=(0, 0, 255), thickness=-1
                )

            cv2.imwrite(
                pool_dir + "{:06d}".format(savedIndex) + "demo.png", original_img,
            )

            global_pred_pose = OM.symmetricRemove(global_pred_pose)

            np.save(
                pool_dir + "{:06d}".format(savedIndex) + "initPose.npy",
                global_pred_pose,
            )

            # copy label mask img
            shutil.copyfile(
                processed_dir + "{:06d}".format(savedIndex) + "labelmask_original.png",
                pool_dir + "{:06d}".format(savedIndex) + "labelmask.png",
            )

            # copy target pose
            shutil.copyfile(
                processed_dir + "{:06d}".format(savedIndex) + "targetPose_original.npy",
                pool_dir + "{:06d}".format(savedIndex) + "targetPose.npy",
            )


# remove the directory
def removeFilesInDir(directory):
    filenames = []
    pool_path = Path(directory)
    for f in pool_path.iterdir():
        os.remove(str(f))


def main():
    global counter
    global output_counter
    # copy all files from dataset to pool
    input_filepath = CFG.REFINE_ITERATIVE_DATA_PATH

    # generate a pool and processed directory if need
    if not os.path.isdir(pool_dir):
        os.makedirs(os.path.dirname(pool_dir), exist_ok=True)
    else:
        # if pool dir exists, then delete all files in it
        removeFilesInDir(pool_dir)

    if not os.path.isdir(processed_dir):
        os.makedirs(os.path.dirname(processed_dir), exist_ok=True)
    else:
        # if processed dir exists, then delete all files in it
        removeFilesInDir(processed_dir)

    # read images and poses
    input_path = Path(input_filepath)
    for f in input_path.iterdir():
        shutil.copyfile(
            str(f), pool_dir + str(f.name),
        )

    learningrate = lr
    numOfTotalIteration = 8

    for iteration in range(numOfTotalIteration):
        # # generate the processed data
        # read images and poses
        input_path = Path(pool_dir)
        image_names, initpose_names, targetpose_names, mask_names = [], [], [], []
        for f in input_path.iterdir():
            if f.match("*img.png"):
                image_names.append(str(f))
            if f.match("*targetPose.npy"):
                targetpose_names.append(str(f))
            if f.match("*mask.png"):
                mask_names.append(str(f))
            if f.match("*initPose.npy"):
                initpose_names.append(str(f))

        image_names.sort()
        targetpose_names.sort()
        mask_names.sort()
        initpose_names.sort()

        # reduce size
        # image_names = image_names[:10]
        # targetpose_names = targetpose_names[:10]
        # mask_names = mask_names[:10]
        # initpose_names = initpose_names[:10]

        datalist = list(zip(image_names, initpose_names, targetpose_names, mask_names,))

        inputP = []
        for o in range(cpu_count()):
            inputP.append((o, list(datalist)))

        # init the counters
        counter.value = 0
        output_counter.value = 0

        # delete all files in temp directory
        removeFilesInDir(processed_dir)

        # run processed data generation parallelly
        with Pool() as p:
            p.imap_unordered(process_data, inputP)
            p.close()
            p.join()

        # update the train.txt and val.txt
        numberOfData = output_counter.value
        val_list = random.sample(range(numberOfData), int((numberOfData + 1) * 0.3))
        train_list = [i for i in range(numberOfData) if i not in val_list]

        f = open(processed_dir + "train.txt", "w")
        for i in train_list:
            f.write("{:06d}".format(i) + "\n")
        f.close()

        f = open(processed_dir + "val.txt", "w")
        for i in val_list:
            f.write("{:06d}".format(i) + "\n")
        f.close()

        # initialize the object
        obj = init(0.00001)

        sample_points = torch.as_tensor(obj.sharp_sample_points)

        # train data set
        # build train data loader
        train_dataset = Iterative_refine_data(data_path=processed_dir, isTrain=True)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
        )

        # build val data loader
        val_dataset = Iterative_refine_data(data_path=processed_dir, isTrain=False)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True, num_workers=8
        )

        print("start training...")

        pre_loss = None
        for epoch in range(epochs):

            tem = train(sample_points, train_loader, train_dataset)
            print("Finish epoch {}, loss {}".format(epoch, tem))
            valtem = val(sample_points, val_loader, val_dataset)

            if pre_loss == None:
                torch.save(mymodel, CFG.BEST_MODEL_ITERATIVE_REFINE)
                pre_loss = valtem
            elif pre_loss > valtem:
                torch.save(mymodel, CFG.BEST_MODEL_ITERATIVE_REFINE)
                pre_loss = valtem
            if (epoch + 1) % 40 == 0:
                scheduler.step()

        print("training process done for interation ", iteration)
        if iteration == numberOfData - 1:
            break

        # # generate next dataset
        # generate the new val file for generate data set
        val_list = [i for i in range(numberOfData)]
        f = open(processed_dir + "val.txt", "w")
        for i in val_list:
            f.write("{:06d}".format(i) + "\n")
        f.close()

        removeFilesInDir(pool_dir)
        obj = init(0.001)
        generateData(obj, sample_points)
        removeFilesInDir(processed_dir)
        print("data generation done!")
        OM.exit()

        # reset the learning rate
        # for g in optimizer.param_groups:
        #     learningrate = 0.4 * learningrate
        #     g["lr"] = learningrate


if __name__ == "__main__":
    main()

