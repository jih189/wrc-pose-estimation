import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.common.DataLoader import PSP_data
from models.model import PSPNet
import numpy as np

batch_size = 64
epochs = 500
lr = 1e-4
class_weights = [0.1, 1.0]

train_dir = "data/processed/pulley_refine/"
val_dir = "data/processed/pulley_refine/"

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

        val_loss = val()

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
    for data in val_loader:
        idx, img, labelmask_img = data

        input = Variable(img).cuda().float()
        label = Variable(labelmask_img).cuda().long()
        label = label.squeeze(1)

        with torch.no_grad():
            output = model(input)
        output = output.squeeze(1)

        loss = seg_criterion(output, label)

        epoch_losses.append(loss.data.cpu().numpy())
    avg_loss = sum(epoch_losses) / len(val_loader)
    return avg_loss


if __name__ == "__main__":
    train()
