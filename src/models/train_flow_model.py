from src.common.DataLoader import FlowNet_data
from models.model import FlowNet
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
import src.configuration as CFG
import cv2

batch_size = 64
epochs = 200
lr = 8e-5
momentum = 0.9
w_decay = 1.0
seglambda = 1000.0

train_dir = CFG.REFINE_DATA_PATH
val_dir = CFG.REFINE_DATA_PATH

# build train data loader
train_dataset = FlowNet_data(data_path=train_dir, isTrain=True)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
)

# build val data loader
val_dataset = FlowNet_data(data_path=val_dir, isTrain=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

# initiate the net
mymodel = FlowNet().cuda()
mymodel = nn.DataParallel(mymodel)

seg_criterion = nn.CrossEntropyLoss(reduce=False)

# wandb.watch(mymodel)

optimizer = optim.Adam(
    mymodel.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-08, weight_decay=w_decay
)


lmbda = lambda epoch: 0.5
scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

torch.autograd.set_detect_anomaly(True)


def train():
    print("start training...")
    pre_loss = None
    for epoch in range(epochs):
        avg_loss = []
        avg_flowloss = []
        avg_segloss = []
        for data in train_loader:
            (idx, img, edge_img, mask_img, labelmask_img, flow_img) = data

            mask_img = mask_img.cuda().float() / 255.0
            edge_img = edge_img.cuda().float() / 255.0
            img = img.cuda().float() / 255.0
            labelflow = flow_img[:, :2, :, :].cuda().float()

            optimizer.zero_grad()

            # catenate image, edges, mask, and bounding box into one input
            # input order (edge, mask, bounding box, image)
            inputData = torch.cat((mask_img, edge_img, img), 1,)

            input = Variable(inputData)

            labelmask_img = Variable(labelmask_img).cuda().long()
            labelmask_img = labelmask_img.squeeze(1)

            ################ test
            # testindex = 0
            # testimg = np.transpose(flow_img[testindex].numpy(), (1, 2, 0)).copy()
            ################

            # predict the rotation, translation, and depth in image view
            opticalFlow, segmentMask, _ = mymodel(input)
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

            flowloss = flowloss.sum()
            loss = flowloss + segloss
            loss.backward()
            optimizer.step()
            avg_loss.append(loss.data.cpu().numpy())
            avg_flowloss.append(flowloss.data.cpu().numpy())
            avg_segloss.append(segloss.data.cpu().numpy())
        tem = sum(avg_loss) / len(train_dataset)
        flow_tem = sum(avg_flowloss) / len(train_dataset)
        seg_tem = sum(avg_segloss) / len(train_dataset)

        print(
            "Finish epoch {}, loss {}, flow loss {} seg loss {}".format(
                epoch, tem, flow_tem, seg_tem
            )
        )

        val_loss, iou = val()
        print("val loss = {} iou = {} ".format(val_loss, iou))

        if pre_loss == None:
            torch.save(mymodel.module.state_dict(), CFG.BEST_MODEL_FLOWNET)
            pre_loss = val_loss
        elif pre_loss > val_loss:
            torch.save(mymodel.module.state_dict(), CFG.BEST_MODEL_FLOWNET)
            pre_loss = val_loss
        mymodel.train()
        if (epoch + 1) % 50 == 0:
            scheduler.step()


def val():
    mymodel.eval()
    avg_loss = []
    avg_iou = []
    for data in val_loader:
        (idx, img, edge_img, mask_img, labelmask_img, flow_img) = data

        mask_img = mask_img.cuda().float() / 255.0
        edge_img = edge_img.cuda().float() / 255.0
        img = img.cuda().float() / 255.0
        labelflow = flow_img[:, :2, :, :].cuda().float()

        inputData = torch.cat((mask_img, edge_img, img), 1,)
        input = Variable(inputData)

        labelmask_img = Variable(labelmask_img).cuda().long()
        labelmask_img = labelmask_img.squeeze(1)

        with torch.no_grad():
            opticalFlow, segmentMask, _ = mymodel(input)

        predictflow = torch.sigmoid(opticalFlow)

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

        flowloss = flowloss.sum()
        loss = flowloss + segloss
        iou_value = iou(labelmask_img, segmentMask, 1)
        avg_loss.append(loss.data.cpu().numpy())
        avg_iou.append(iou_value.data.cpu().numpy())
    tem = sum(avg_loss) / len(val_dataset)
    ioutem = sum(avg_iou) / len(val_dataset)
    return tem, ioutem


if __name__ == "__main__":

    train()
    print("done")
