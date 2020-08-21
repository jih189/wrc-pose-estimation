import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.common.DataLoader import PSP_data
from src.common.iou import iou
from models.model import PSPNet
import numpy as np
import src.configuration as CFG

batch_size = 64
epochs = 500
lr = 5e-3
class_weights = [0.1, 1.0]

# train_dir = "data/processed/pulley_refine/"
# val_dir = "data/processed/pulley_refine/"

train_dir = CFG.REFINE_SATA_PATH
val_dir = CFG.REFINE_SATA_PATH

train_dataset = PSP_data(data_path=train_dir, isTrain=True)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=12
)

val_dataset = PSP_data(data_path=val_dir, isTrain=False)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=12
)

model = PSPNet().cuda()
model = nn.DataParallel(model)

model.train()

seg_criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=lr)
lmbda = lambda epoch: 0.5
scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)


def train():
    print("start training:")
    pre_loss = None
    for epoch in range(epochs):
        epoch_losses = []
        for data in train_loader:
            idx, img, labelmask_img = data

            optimizer.zero_grad()

            input = Variable(img).cuda().float()
            label = Variable(labelmask_img).cuda().long()
            label = label.squeeze(1)

            output = model(input)
            output = output.squeeze(1)

            loss = seg_criterion(output, label)

            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.data.cpu().numpy())

        avg_loss = sum(epoch_losses) / len(train_dataset)

        print("Finish epoch {},loss {}".format(epoch, avg_loss))

        val_loss, val_iou = val()
        print("val loss {} and iou {}".format(val_loss, val_iou))

        if pre_loss == None:
            torch.save(model, "best_model_psp.pth")
            pre_loss = val_loss
        elif pre_loss > val_loss:
            torch.save(model, "best_model_psp.pth")
            pre_loss = val_loss

        model.train()

        if (epoch + 1) % 50 == 0:
            scheduler.step()


def val():
    model.eval()
    epoch_losses = []
    epoch_ious = []
    for data in val_loader:
        idx, img, labelmask_img = data

        input = Variable(img).cuda().float()
        label = Variable(labelmask_img).cuda().long()
        label = label.squeeze(1)

        with torch.no_grad():
            output = model(input)
        output = output.squeeze(1)

        loss = seg_criterion(output, label)
        iou_value = iou(label, output, 1)

        epoch_losses.append(loss.data.cpu().numpy())
        epoch_ious.append(iou_value.data.cpu().numpy())
    avg_loss = sum(epoch_losses) / len(val_loader)
    avg_iou = sum(epoch_ious) / len(val_loader)
    return avg_loss, avg_iou


if __name__ == "__main__":
    train()
