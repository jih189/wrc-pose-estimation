# run testing on image
import numpy as np
import cv2
import matplotlib.pyplot as plt

from models.model import FlowNet
import kornia

import torch
import torch.nn as nn

from torch.autograd import Variable
import src.configuration as CFG

IMG_SIZE = 240


def init():

    mymodel = FlowNet().cuda()

    mymodel = nn.DataParallel(mymodel)
    mymodel = torch.load("best_model_flownet.pth")
    mymodel.eval()

    return mymodel


def predict(mymodel, predict_index, view_image):
    numForTest = "{:06d}".format(predict_index)
    processed_data_dir = CFG.REFINE_SATA_PATH

    # load rgb image
    img_path = processed_data_dir + numForTest + "img.png"
    img = cv2.imread(img_path)

    # calculate the resize scale
    rescaleValue = float(IMG_SIZE) / img.shape[0]

    # resize rgb image
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    testimg = img.copy()
    preimg = img.copy()
    img = img[:, :, :3].transpose(2, 0, 1)

    img = Variable(torch.from_numpy(img).cuda()).float()
    img = img.unsqueeze(0)

    # load edge image
    edge_path = processed_data_dir + numForTest + "edge.png"
    edge_img = cv2.imread(edge_path)
    edge_img = cv2.resize(edge_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    edge_img = edge_img[:, :, :1].transpose(2, 0, 1)

    edge_img = Variable(torch.from_numpy(edge_img).cuda()).float()
    edge_img = edge_img.unsqueeze(0)

    # load the mask image
    mask_path = processed_data_dir + numForTest + "mask.png"
    mask_img = cv2.imread(mask_path)
    mask_img = cv2.resize(mask_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    mask_img = mask_img[:, :, :1].transpose(2, 0, 1)

    mask_img = Variable(torch.from_numpy(mask_img).cuda()).float()
    mask_img = mask_img.unsqueeze(0)

    # load the flow image
    flow_path = processed_data_dir + numForTest + "flow.png"
    flow_img = cv2.imread(flow_path)
    flow_img = cv2.resize(flow_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # running model
    inputData = torch.cat(
        (mask_img.cuda().float() / 255.0, edge_img.cuda().float() / 255.0, img.cuda().float() / 255.0,), 1,
    )
    input = Variable(inputData)
    output, segoutput, _ = mymodel(input)
    predictflow = torch.sigmoid(output)

    padding = Variable(
        torch.zeros(predictflow.shape[0], 1, predictflow.shape[2], predictflow.shape[3])
    ).cuda()

    predictflow = torch.cat((predictflow, padding), 1)

    mask = mask_img == 255.0

    predictflow = predictflow * mask

    edge_test = cv2.imread(edge_path)
    edge_test = cv2.resize(
        edge_test, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
    )

    oriimg = cv2.imread(img_path)
    oriimg = cv2.resize(oriimg, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    invflow = np.zeros(oriimg.shape)


    seg_pred = torch.argmax(segoutput, 1, keepdim=True).float().squeeze(1).cpu().detach().numpy()

    cv2.imshow("mask", seg_pred[0])

    for y in range(flow_img.shape[0]):
        for x in range(flow_img.shape[1]):
            if seg_pred[0,y,x] == 1:
                invflow[y, x] = [0,255,0]
    for y in range(flow_img.shape[0]):
        for x in range(flow_img.shape[1]):
            if edge_test[y, x, 0] == 255.0:
                oriimg = cv2.circle(
                    oriimg, (x, y), radius=0, color=(0, 0, 255), thickness=-1,
                )
            # if x % 10 == 0 and y % 10 == 0:
            [mx, my] = predictflow[0, :2, y, x].cpu().detach().numpy()
            if mx != 0.0 or my != 0.0:
                mx = int((mx - 0.5) * IMG_SIZE)
                my = int((my - 0.5) * IMG_SIZE)
                if x + mx >= 0 and x + mx < oriimg.shape[0] and y + my >= 0 and y + my < oriimg.shape[1]:
                    invflow = cv2.circle(
                        invflow,
                        (x + mx, y + my),
                        radius=0,
                        color=(255, 255, 255),
                        thickness=-1,
                    )

                    if x % 20 == 0 and y % 20 == 0:

                        oriimg = cv2.line(oriimg, (x, y), (x + mx, y + my), color=(0,255,0), thickness = 1)
                        oriimg = cv2.circle(
                            oriimg,
                            (x + mx, y + my),
                            radius=0,
                            color=(255, 0, 0),
                            thickness=-1,
                        )

                

    testimg = np.transpose(predictflow[0].cpu().detach().numpy(), (1, 2, 0)).copy()
    oriimg = cv2.resize(
        oriimg, (IMG_SIZE * 5, IMG_SIZE * 5), interpolation=cv2.INTER_AREA
    )
    cv2.imshow("inv flow", invflow)
    cv2.imshow("test", testimg)
    cv2.imshow("label", flow_img)
    cv2.imshow("result", oriimg)
    cv2.waitKey(0)


if __name__ == "__main__":
    m = init()
    predict(m, 3176, True)
