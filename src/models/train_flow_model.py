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
epochs = 1000
lr = 3e-4
momentum = 0.9
w_decay = 2.0

train_dir = CFG.REFINE_SATA_PATH
val_dir = CFG.REFINE_SATA_PATH

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

seg_criterion = nn.CrossEntropyLoss()

# wandb.watch(mymodel)

optimizer = optim.Adam(
    mymodel.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-08, weight_decay=w_decay
)


lmbda = lambda epoch: 0.5
scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

torch.autograd.set_detect_anomaly(True)


def train():
    print("start training... Nahid habibi")
    pre_loss = None
    for epoch in range(epochs):
        avg_loss = []
        avg_shiftloss = []
        avg_segloss = []
        for data in train_loader:
            (idx, img, edge_img, mask_img, labelmask_img, flow_img) = data

            optimizer.zero_grad()

            # catenate image, edges, mask, and bounding box into one input
            # input order (edge, mask, bounding box, image)
            inputData = torch.cat(
                (mask_img.cuda().float(), edge_img.cuda().float(), img.cuda().float(),),
                1,
            )
            input = Variable(inputData)

            label = Variable(labelmask_img).cuda().long()
            label = label.squeeze(1)

            ################ test
            # testindex = 0
            # testimg = np.transpose(flow_img[testindex].numpy(), (1, 2, 0)).copy()
            ################

            # predict the rotation, translation, and depth in image view
            output, psp_out = mymodel(input)
            predictflow = torch.sigmoid(output)
            labelflow = flow_img[:, :2, :, :].cuda()

            psp_out = psp_out.squeeze(1)

            segloss = seg_criterion(psp_out, label) * 1000

            # invalid flow is defined with both flow coordinates to be exactly 0
            mask = (labelflow[:, 0] == 0) & (labelflow[:, 1] == 0)

            shiftLoss = torch.norm(predictflow - labelflow, 2, 1)

            shiftLoss = shiftLoss[~mask]
            shiftLoss = shiftLoss.sum() / labelflow.size(0)
            loss = shiftLoss + segloss
            loss.backward()
            optimizer.step()
            avg_loss.append(loss.data.cpu().numpy())
            avg_shiftloss.append(shiftLoss.data.cpu().numpy())
            avg_segloss.append(segloss.data.cpu().numpy())
        tem = sum(avg_loss) / len(train_dataset)
        shift_tem = sum(avg_shiftloss) / len(train_dataset)
        seg_tem = sum(avg_segloss) / len(train_dataset)
        print(
            "Finish epoch {}, loss {}, flow loss {} seg loss {}".format(
                epoch, tem, shift_tem, seg_tem
            )
        )

        val_loss, iou = val()
        print("val loss = {} iou = {} ".format(val_loss, iou))

        if pre_loss == None:
            torch.save(mymodel, "best_model_flownet.pth")
            pre_loss = val_loss
        elif pre_loss > val_loss:
            torch.save(mymodel, "best_model_flownet.pth")
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

        inputData = torch.cat(
            (mask_img.cuda().float(), edge_img.cuda().float(), img.cuda().float(),), 1,
        )
        input = Variable(inputData)

        label = Variable(labelmask_img).cuda().long()
        label = label.squeeze(1)

        with torch.no_grad():
            output, psp_out = mymodel(input)

        predictflow = torch.sigmoid(output)
        labelflow = flow_img[:, :2, :, :].cuda()

        psp_out = psp_out.squeeze(1)

        segloss = seg_criterion(psp_out, label) * 1000

        iou_value = iou(label, psp_out, 1)

        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (labelflow[:, 0] == 0) & (labelflow[:, 1] == 0)

        shiftLoss = torch.norm(predictflow - labelflow, 2, 1)

        shiftLoss = shiftLoss[~mask]
        # confidLoss = confidLoss[~mask]
        shiftLoss = shiftLoss.sum() / labelflow.size(0)
        loss = shiftLoss + segloss
        avg_loss.append(loss.data.cpu().numpy())
        avg_iou.append(iou_value.data.cpu().numpy())
    tem = sum(avg_loss) / len(val_dataset)
    ioutem = sum(avg_iou) / len(val_loader)
    return tem, ioutem


if __name__ == "__main__":

    train()
    print("done")
