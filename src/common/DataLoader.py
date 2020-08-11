import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import src.common.object_model as OM
import src.configuration as CFG
from torchvision import utils
import random

numberOfSampledPoint = 1000


class Rot_data(Dataset):
    def __init__(
        self,
        n_class=200,
        data_path="",
        pose_data_path="",
        resX=240,
        resY=240,
        isTrain=True,
    ):
        trainBit = ""
        if isTrain:
            trainOrVal = "train.txt"
        else:
            trainOrVal = "val.txt"
        self.n_class = n_class
        self.dataNames = []
        self.dataPath = data_path
        self.poseDataPath = pose_data_path
        with open(data_path + trainOrVal, "r") as reader:
            # read the data path
            for line in reader.readlines():
                self.dataNames.append(line.rstrip("\n"))
        self.resX = resX
        self.resY = resY

        OM.setup(CFG.CAMERA_W, CFG.CAMERA_H)

        OM.setProjectMatrixWithIntr(CFG.CAMERA_MATRIX, CFG.CAMERA_W, CFG.CAMERA_H)

        self.obj = OM.ObjectModel()
        self.obj.loadObjectCADModel(CFG.CAD_MODEL)
        self.obj.setIntrinsicMatrix(CFG.CAMERA_MATRIX)

        self.obj.determineSharpEdges(0.05)
        self.obj.generateSamplePoints(0.001, 0.001)

    def __len__(self):
        return len(self.dataNames)

    def __getitem__(self, idx):
        # create label
        label_path = self.poseDataPath + self.dataNames[idx] + ".npy"
        pose = np.load(label_path)
        # remove the symmetry
        pose = OM.symmetricRemove(pose)
        self.obj.setModelviewMatrix(pose)
        viewPoint, inplaneRotation, offsetFromCenter, depth = self.obj.getLabel()
        vpidx = OM.cal_idx(viewPoint)

        center_pt = self.obj.project3Dto2D((0, 0, 0), pose)

        # (upperleftx, upperlefty, lowerrightx, lowerrighty)
        boundingbox = np.load(self.dataPath + "bounding" + self.dataNames[idx] + ".npy")
        upperleftx, upperlefty, lowerrightx, lowerrighty = (
            boundingbox[0].astype(np.int),
            boundingbox[1].astype(np.int),
            boundingbox[2].astype(np.int),
            boundingbox[3].astype(np.int),
        )

        l = lowerrightx - upperleftx

        offset = np.array(
            [(center_pt[0] - upperleftx) / l, (center_pt[1] - upperlefty) / l]
        )

        img_path = self.dataPath + "crop" + self.dataNames[idx] + ".png"
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.resY, self.resX), interpolation=cv2.INTER_AREA)
        img = img[:, :, :3].transpose(2, 0, 1)

        inplaneRotation = inplaneRotation % (2 * np.pi) / (2 * np.pi / 60)
        inplaneRotation = int(inplaneRotation)

        return img, vpidx, inplaneRotation, offset, depth


class Refine_data(Dataset):
    def __init__(self, data_path="", isTrain=True):
        trainBit = ""
        if isTrain:
            trainOrVal = "train.txt"
        else:
            trainOrVal = "val.txt"
        self.dataNames = []
        with open(data_path + trainOrVal, "r") as reader:
            for line in reader.readlines():
                self.dataNames.append(data_path + line.rstrip("\n"))

        self.imgSize = 240

    def __len__(self):
        return len(self.dataNames)

    def __getitem__(self, idx):
        # create label
        # load rgb image
        img_path = self.dataNames[idx] + "img.png"
        img = cv2.imread(img_path)

        # calculate the resize scale
        rescaleValue = float(self.imgSize) / img.shape[1]

        # resize rgb image
        img = cv2.resize(
            img, (self.imgSize, self.imgSize), interpolation=cv2.INTER_AREA
        )
        img = img[:, :, :3].transpose(2, 0, 1)

        # load edge image
        edge_path = self.dataNames[idx] + "edge.png"
        edge_img = cv2.imread(edge_path)
        edge_img = cv2.resize(
            edge_img, (self.imgSize, self.imgSize), interpolation=cv2.INTER_AREA
        )
        edge_img = edge_img[:, :, :1].transpose(2, 0, 1)

        # load bounding image
        # bounding_path = self.dataNames[idx] + "bounding.png"
        # bounding_img = cv2.imread(bounding_path)
        # bounding_img = cv2.resize(
        #     bounding_img, (self.imgSize, self.imgSize), interpolation=cv2.INTER_AREA
        # )
        # bounding_img = bounding_img[:, :, :1].transpose(2, 0, 1)

        # load the mask image
        mask_path = self.dataNames[idx] + "mask.png"
        mask_img = cv2.imread(mask_path)
        mask_img = cv2.resize(
            mask_img, (self.imgSize, self.imgSize), interpolation=cv2.INTER_AREA
        )
        mask_img = mask_img[:, :, :1].transpose(2, 0, 1)

        # load the init 3d points
        init3dPt_path = self.dataNames[idx] + "init3dPt.npy"
        init3dPt = np.load(init3dPt_path)

        # sample points
        init3dPtRamdomIdx = np.random.choice(
            init3dPt.shape[0], numberOfSampledPoint, replace=False
        )
        init3dPt = init3dPt[init3dPtRamdomIdx]

        # load init pose
        initPose_path = self.dataNames[idx] + "initPose.npy"
        initPose = np.load(initPose_path)

        # load the target 3d points
        target3dPt_path = self.dataNames[idx] + "target3dPt.npy"
        target3dPt = np.load(target3dPt_path)
        # sample points
        target3dPtRandomIdx = np.random.choice(
            target3dPt.shape[0], numberOfSampledPoint, replace=False
        )
        target3dPt = target3dPt[target3dPtRandomIdx]

        # load the target pose
        targetPose_path = self.dataNames[idx] + "targetPose.npy"
        targetPose = np.load(targetPose_path)

        return (
            idx,
            img,
            edge_img,
            mask_img,
            init3dPt,
            initPose,
            target3dPt,
            targetPose,
            rescaleValue,
        )


class PSP_data(Dataset):
    def __init__(self, data_path="", isTrain=True):
        trainBit = ""
        if isTrain:
            trainOrVal = "train.txt"
        else:
            trainOrVal = "val.txt"
        self.dataNames = []
        with open(data_path + trainOrVal, "r") as reader:
            for line in reader.readlines():
                self.dataNames.append(data_path + line.rstrip("\n"))

        self.imgSize = 240

    def __len__(self):
        return len(self.dataNames)

    def __getitem__(self, idx):
        # create label
        # load rgb image
        img_path = self.dataNames[idx] + "img.png"
        img = cv2.imread(img_path)

        # resize rgb image
        img = cv2.resize(
            img, (self.imgSize, self.imgSize), interpolation=cv2.INTER_AREA
        )
        img = img[:, :, :3].transpose(2, 0, 1)

        # load label mask image
        labelmask_path = self.dataNames[idx] + "labelmask.png"
        labelmask_img = cv2.imread(labelmask_path)
        labelmask_img = cv2.resize(
            labelmask_img, (self.imgSize, self.imgSize), interpolation=cv2.INTER_AREA
        )
        labelmask_img = labelmask_img[:, :, :1].transpose(2, 0, 1) / 255.0
        return (
            idx,
            img,
            labelmask_img,
        )
