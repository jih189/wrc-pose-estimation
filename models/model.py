import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# from spatial_correlation_sampler import SpatialCorrelationSampler


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class Magic_Net(nn.Module):
    def __init__(self, viewpt_class=7, rot_class=10):
        super(Magic_Net, self).__init__()
        self.viewpt_class = viewpt_class
        self.rot_class = rot_class

        self.resnet = models.resnet50(pretrained=True)
        self.resnet.layer4[1].conv2 = nn.Conv2d(
            512, 512, kernel_size=15, stride=1, padding=7
        )

        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.resnet.fc = nn.Linear(2048, self.viewpt_class + self.rot_class + 2)

    def forward(self, x):
        x1 = self.resnet(x)
        return x1


class Refine_Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = models.resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(
            6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.resnet.layer4[1].conv2 = nn.Conv2d(
            512, 512, kernel_size=15, stride=1, padding=7
        )

        # self.dropout = nn.Dropout2d(p=0.1)

        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.resnet.fc = nn.Linear(2048, 512)

        self.fcrotation = nn.Linear(512, 3)
        self.fctraslation = nn.Linear(512, 2)
        self.fcdist = nn.Linear(512, 1)
        self.distPlus = nn.Softplus(threshold=1.0)

        # flownet based with two branchs

        # self.correlation_sampler = SpatialCorrelationSampler(1, 9, 1, 0, 2)

        # # render
        # self.conv0_r = nn.Conv2d(2, 64, kernel_size=5, stride=1, padding=2)
        # self.bn0_r = nn.BatchNorm2d(64)
        # self.m0_r = Refine_Net.create_stage(64)

        # # img
        # self.conv0_i = nn.Conv2d(4, 64, kernel_size=5, stride=1, padding=2)
        # self.bn0_i = nn.BatchNorm2d(64)
        # self.m0_i = Refine_Net.create_stage(64)

        # self.maxpool = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(128)
        # self.m1 = Refine_Net.create_stage(128)

        # self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(256)
        # self.m2 = Refine_Net.create_stage(256)

        # self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(512)
        # self.m3 = Refine_Net.create_stage(512)

        # self.conv4 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        # self.bn4 = nn.BatchNorm2d(1024)
        # self.m4 = Refine_Net.create_stage(1024)

        # self.conv5 = nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1)
        # self.bn5 = nn.BatchNorm2d(2048)
        # self.m5 = Refine_Net.create_stage(2048)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.fc = nn.Linear(2048, 512)
        # self.fc2 = nn.Linear(512, 256)

        # self.fcrotation = nn.Linear(256, 3)
        # self.fctraslation = nn.Linear(256, 2)
        # self.fcdist = nn.Linear(256, 1)
        # self.distPlus = nn.Softplus(threshold=1.0)

    def forward(self, x):
        # x1_r = self.bn0_r(self.relu(self.conv0_r(x[:, 0:2])))
        # x2_r = self.maxpool(x1_r + self.m0_r(x1_r))

        # x1_i = self.bn0_i(self.relu(self.conv0_i(x[:, 2:6])))
        # x2_i = self.maxpool(x1_i + self.m0_i(x1_i))

        # corr = self.correlation_sampler(x2_r, x2_i)
        # corr1 = corr.view(
        #     corr.shape[0], corr.shape[1] * corr.shape[2], corr.shape[3], corr.shape[4]
        # )

        # corr2 = self.relu(corr1)

        # print("corr", corr.shape)

        # x1 = self.bn0(self.relu(self.conv0(x)))
        # x2 = self.maxpool(x1 + self.m0(x1))

        # x3 = self.bn1(self.relu(self.conv1(x2)))
        # x4 = self.maxpool(x3 + self.m1(x3))

        # x5 = self.bn2(self.relu(self.conv2(x4)))
        # x6 = self.maxpool(x5 + self.m2(x5))

        # x7 = self.bn3(self.relu(self.conv3(x6)))
        # x8 = self.maxpool(x7 + self.m3(x7))

        # x7_1 = self.bn4(self.relu(self.conv4(x8)))
        # x8_1 = self.maxpool(x7_1 + self.m4(x7_1))

        # x7_2 = self.bn5(self.relu(self.conv5(x8_1)))
        # x8_2 = self.maxpool(x7_2 + self.m5(x7_2))

        # x9 = torch.reshape(self.avgpool(x8_2), (-1, 2048))

        # x9_1 = self.fc(x9)
        # x10 = self.fc2(x9_1)
        # x11 = self.fcrotation(x10)
        # x12 = self.fctraslation(x10)
        # x13 = self.fcdist(x10)
        # x14 = self.distPlus(x13)

        x2 = self.resnet(x)
        x11 = self.fcrotation(x2)
        x12 = self.fctraslation(x2)
        x13 = self.fcdist(x2)
        x14 = self.distPlus(x13)
        x15 = torch.sigmoid(x12)
        return x11, x15, x14

    @staticmethod
    def create_stage(out_channels):
        kernel = 3
        padding = 1
        num_feature = 32
        model = nn.Sequential()
        model.add_module(
            "conv0",
            nn.Conv2d(out_channels, num_feature, kernel_size=1, stride=1, padding=0),
        )
        model.add_module("relu0", nn.ReLU(inplace=True))
        model.add_module("bnd0", nn.BatchNorm2d(num_feature))
        model.add_module(
            "conv1",
            nn.Conv2d(
                num_feature, num_feature, kernel_size=kernel, stride=1, padding=padding
            ),
        )
        model.add_module("relu1", nn.ReLU(inplace=True))
        model.add_module("bnd1", nn.BatchNorm2d(num_feature))
        model.add_module(
            "conv2",
            nn.Conv2d(num_feature, out_channels, kernel_size=1, stride=1, padding=0),
        )
        model.add_module("relu2", nn.ReLU(inplace=True))
        model.add_module("bnd2", nn.BatchNorm2d(out_channels))
        return model


class PSPNet(nn.Module):
    def __init__(self):
        super(PSPNet, self).__init__()
        num_classes = 2
        resnet = models.resnet50(pretrained=True)
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.decov = nn.ConvTranspose2d(2048, 1024, 3, 2, 1)
        self.final = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=0.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1),
        )

    def forward(self, x):
        x_size = x.size()
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.decov(self.layer4(x3))
        x5 = torch.cat([x3, x4], 1)

        x6 = self.final(x5)
        return F.interpolate(x6, x_size[2:], mode="bilinear", align_corners=False)
