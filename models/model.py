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

        self.layer4[2].conv2 = nn.Conv2d(512, 512, kernel_size=15, stride=1, padding=7)

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


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=True,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=False)


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, : target.size(2), : target.size(3)]


class FlowNet(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True):
        super(FlowNet, self).__init__()

        num_classes = 2

        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, 5, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        self.decov6 = nn.ConvTranspose2d(1024, 512, 15, 15, 0)
        self.decov4 = nn.ConvTranspose2d(512, 256, 4, 4, 0)
        self.decov2 = nn.ConvTranspose2d(128, 256, 3, 1, 1)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.combineFlow = nn.Conv2d(10, 2, kernel_size=1, stride=1, padding=0)
        self.final = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(512, momentum=0.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1),
        )

    def forward(self, x):
        x_size = x.size()

        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)

        out = [
            F.interpolate(flow2, x_size[2:], mode="bilinear", align_corners=False),
            F.interpolate(flow3, x_size[2:], mode="bilinear", align_corners=False),
            F.interpolate(flow4, x_size[2:], mode="bilinear", align_corners=False),
            F.interpolate(flow5, x_size[2:], mode="bilinear", align_corners=False),
            F.interpolate(flow6, x_size[2:], mode="bilinear", align_corners=False),
        ]

        psp6 = self.decov6(out_conv6)
        psp4 = self.decov4(out_conv4)
        psp2 = self.decov2(out_conv2)
        psp = torch.cat([psp2, psp4, psp6], 1)
        psp = self.final(psp)

        psp_out = F.interpolate(psp, x_size[2:], mode="bilinear", align_corners=False)

        out = torch.cat(out, 1)
        out = self.combineFlow(out)
        return out, psp_out

