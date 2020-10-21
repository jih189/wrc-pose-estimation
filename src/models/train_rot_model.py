import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

from src.common.DataLoader import Rot_data
from models.model import Magic_Net
import src.configuration as CFG
import numpy as np

# ignore warming
np.seterr(divide="ignore", invalid="ignore")

# import wandb

# wandb.init(project="wrc-rot-classifier")

batch_size = 64
epochs = 1000
lr = 3e-5
momentum = 0.9
w_decay = 4e-3
viewpt_class = 64
rot_class = 60

train_dir = CFG.PROCESSED_DATA_PATH
val_dir = CFG.PROCESSED_DATA_PATH
raw_dir = CFG.PROCESSED_DATA_PATH

train_dataset = Rot_data(data_path=train_dir, pose_data_path=raw_dir, isTrain=True)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=12
)

val_dataset = Rot_data(data_path=val_dir, pose_data_path=raw_dir, isTrain=False)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=12
)

model = Magic_Net(viewpt_class=viewpt_class, rot_class=rot_class).cuda()
model = nn.DataParallel(model)

# Evaluation setup
# model = torch.load(CFG.BEST_MODEL_ROT)
# model.eval()

# read the weights
vp_weights = np.load(CFG.PROCESSED_DATA_PATH + "vp_weight.npy")
rot_weights = np.load(CFG.PROCESSED_DATA_PATH + "rot_weight.npy")

vp_class_weights = torch.FloatTensor(vp_weights).cuda()
viewpt_criterion = nn.CrossEntropyLoss(weight=vp_class_weights)
rot_class_weights = torch.FloatTensor(rot_weights).cuda()
rot_criterion = nn.CrossEntropyLoss(weight=rot_class_weights)
offset_criterion = nn.MSELoss(reduction="sum")
rot_lamda = 5.0
offset_lamda = 0.5

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
lmbda = lambda epoch: 0.5
scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)


def cal_rot_loss(thetas, target):
    return torch.sum(1.0 - torch.cos(thetas - target))


def train():
    print("start training:")
    model.train()
    pre_loss = None
    for epoch in range(epochs):
        avg_loss = []
        for data in train_loader:
            (input_img, vpidx, inplane_rot, offset_label, depth,) = data

            optimizer.zero_grad()

            input = Variable(input_img.cuda()).float()

            output = model(input)

            offset_label = Variable(offset_label.reshape((-1, 2)).cuda()).float()
            vpidx_label = Variable(vpidx.cuda()).long()
            # inplane_rot = Variable(inplane_rot.reshape((-1,1)).cuda()).float()
            inplane_rot = Variable(inplane_rot.cuda()).long()

            # viewpt_labels = vpidx_label.type(dtype = torch.cuda.LongTensor)

            viewpt_loss = viewpt_criterion(output[:, :viewpt_class], vpidx_label)
            rot_loss = rot_criterion(
                output[:, viewpt_class : viewpt_class + rot_class], inplane_rot
            )
            # rot_loss = cal_rot_loss(output[:,viewpt_class:viewpt_class + rot_class], inplane_rot)
            offset_loss = offset_criterion(
                torch.sigmoid(output[:, viewpt_class + rot_class :]), offset_label
            )

            loss = viewpt_loss + rot_loss * rot_lamda + offset_lamda * offset_loss

            loss.backward()
            optimizer.step()
            avg_loss.append(loss.data.cpu().numpy())

        tem = sum(avg_loss) / len(train_dataset)

        print("Finish epoch {},loss {}".format(epoch, tem))

        val_loss, val_rot_loss, val_offset_loss, view_acc, rot_acc = val()

        # wandb.log(
        #     {
        #         "train loss": tem,
        #         "val loss": val_loss,
        #         "val offset loss": val_offset_loss,
        #         "val rot loss": val_rot_loss,
        #         "view point acc": view_acc,
        #         "rotation acc": rot_acc,
        #     }
        # )
        if pre_loss == None:
            torch.save(model, CFG.BEST_MODEL_ROT)
            pre_loss = val_loss
        elif pre_loss > val_loss:
            torch.save(model, CFG.BEST_MODEL_ROT)
            pre_loss = val_loss

        model.train()

        if (epoch + 1) % 90 == 0:
            scheduler.step()


def val():
    model.eval()
    avg_loss = []
    acc_label = 0
    total_label = 0
    acc_rot = 0
    correct_k = 0
    rot_avg_loss = []
    offset_avg_loss = []
    for data in val_loader:
        input_img, vpidx, inplane_rot, offset_label, depth = data

        input = Variable(input_img.cuda()).float()

        with torch.no_grad():
            output = model(input)

        offset_label = Variable(offset_label.reshape((-1, 2)).cuda()).float()
        vpidx_label = Variable(vpidx.cuda()).long()
        inplane_rot = Variable(inplane_rot.cuda()).long()

        # viewpt_labels = vpidx_label.type(dtype = torch.cuda.LongTensor)

        viewpt_loss = viewpt_criterion(output[:, :viewpt_class], vpidx_label)
        rot_loss = rot_criterion(
            output[:, viewpt_class : viewpt_class + rot_class], inplane_rot
        )
        # rot_loss = cal_rot_loss(output[:,viewpt_class:viewpt_class + rot_class], inplane_rot)
        offset_loss = offset_criterion(
            torch.sigmoid(output[:, viewpt_class + rot_class :]), offset_label
        )

        loss = viewpt_loss + rot_loss * rot_lamda + offset_loss * offset_lamda

        pred = output[:, :viewpt_class].data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        acc_label += np.sum(pred == vpidx_label.data.cpu().numpy())
        total_label += len(pred)

        pred = output[:, viewpt_class : viewpt_class + rot_class].data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        acc_rot += np.sum(pred == inplane_rot.data.cpu().numpy())

        avg_loss.append(loss.data.cpu().numpy())
        rot_avg_loss.append(rot_loss.data.cpu().numpy())
        offset_avg_loss.append(offset_loss.cpu().numpy())

        _, idx = output[:, viewpt_class : viewpt_class + rot_class].topk(
            5, 1, largest=True, sorted=True
        )
        idx = idx.t()
        correct = idx.eq(inplane_rot.view(1, -1).expand_as(idx))
        correct_k += correct[:5].view(-1).float().sum(0).data.cpu().numpy()

    tem = sum(avg_loss) / len(val_dataset)
    print("val viewpt acc : ", acc_label / total_label)
    print("val rot acc : ", acc_rot / total_label)
    print("Top 5 rot acc:", correct_k / total_label)

    print("val loss {}".format(tem))
    print("rot loss {}".format(sum(rot_avg_loss) / len(val_dataset)))
    print("offset loss {}".format(sum(offset_avg_loss) / len(val_dataset)))
    return (
        tem,
        sum(rot_avg_loss) / len(val_dataset),
        sum(offset_avg_loss) / len(val_dataset),
        acc_label / total_label,
        acc_rot / total_label,
    )


if __name__ == "__main__":
    train()
    # val()
