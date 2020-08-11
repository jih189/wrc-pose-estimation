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


# class PyramidPoolingModule(nn.Module):
#     def __init__(self, in_dim, reduction_dim, setting):
#         self.features = []
#         for s in setting:
#             self.features.append(nn.Sequential(
#                 nn.AdaptiveAvgPool2d(s)
#                 nn.Conv2d(in_dim, reduction_dim, kernel_size=1,bias=False),
#                 nn.BatchNorm2d(reduction_dim, momentum=0.95),
#                 nn.ReLU(inplace=True)
#             ))

#     def forward(self, x):
#         x_size = x.size()
#         out = [x]
#         for f in self.features:
#             out.append(F.interpolate(f(x), x_size[2:], mode="bilinear", align_corners=False))
#         out = torch.cat(out, 1)
#         return out


class Magic_Net(nn.Module):
    def __init__(self, viewpt_class=64, rot_class=60):
        super(Magic_Net, self).__init__()
        self.viewpt_class = viewpt_class
        self.rot_class = rot_class

        resnet = models.resnet34(pretrained=True)
        # self.resnet.layer4[1].conv2 = nn.Conv2d(
        #     512, 512, kernel_size=15, stride=1, padding=7
        # )
        # self.resnet.layer4[2].conv2 = nn.Conv2d(
        #     512, 512, kernel_size=15, stride=1, padding=7
        # )

        # self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.resnet.fc = nn.Linear(2048, self.viewpt_class + self.rot_class + 2)

        
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        self.layer4[2].conv2 = nn.Conv2d(
            512, 512, kernel_size=15, stride=1, padding=7
        )

        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, self.viewpt_class + self.rot_class + 2)

    def forward(self, x):
        # x = self.resnet(x)
        # return x
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool1(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Refine_Net(nn.Module):
    def __init__(self):
        super().__init__()

        # resnet = models.resnet34(pretrained=True)

        # # rendering branch
        # self.layer0_r = nn.Sequential(
        #     resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        # )
        # self.layer0_r[0] = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.layer1_r, self.layer2_r, self.layer3_r = (
        #     resnet.layer1,
        #     resnet.layer2,
        #     resnet.layer3
        # )

        # # image branch
        # self.layer0_i = nn.Sequential(
        #     resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        # )
        # self.layer0_i[0] = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.layer1_i, self.layer2_i, self.layer3_i = (
        #     resnet.layer1,
        #     resnet.layer2,
        #     resnet.layer3
        # )

        # self.convdown = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        # self.layer4 = resnet.layer4
        # self.layer4[2].conv2 = nn.Conv2d(512, 512, kernel_size=15, stride=1, padding=7)

        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(512, 512)

        self.resnet = models.resnet34(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(
            6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.resnet.layer4[2].conv2 = nn.Conv2d(
            512, 512, kernel_size=15, stride=1, padding=7
        )

        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.resnet.fc = nn.Linear(512, 512)

        self.fcrotation = nn.Linear(512, 3)
        self.fctraslation = nn.Linear(512, 2)
        self.fcdist = nn.Linear(512, 1)
        self.distPlus = nn.Softplus(threshold=1.0)

    def forward(self, x):
        # x_r = self.layer0_r(x[:,:2,:,:])
        # x_r = self.layer1_r(x_r)
        # x_r = self.layer2_r(x_r)
        # x_r = self.layer3_r(x_r)

        # x_i = self.layer0_i(x[:,2:,:,:])
        # x_i = self.layer1_i(x_i)
        # x_i = self.layer2_i(x_i)
        # x_i = self.layer3_i(x_i)

        # x_c = torch.cat([x_r, x_i],1)
        # x_c = self.convdown(x_c)
        # x_c = self.layer4(x_c)
        # x_c = self.avgpool(x_c)
        # x_c = torch.flatten(x_c, 1)
        # x_c = self.fc(x_c)

        # x11 = self.fcrotation(x_c)
        # x12 = self.fctraslation(x_c)
        # x13 = self.fcdist(x_c)
        # x14 = self.distPlus(x13)
        # x15 = torch.sigmoid(x12)
        # return x11, x15, x14

        x2 = self.resnet(x)
        x11 = self.fcrotation(x2)
        x12 = self.fctraslation(x2)
        x13 = self.fcdist(x2)
        x14 = self.distPlus(x13)
        x15 = torch.sigmoid(x12)
        return x11, x15, x14


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
