import numpy as np
import cv2
import torch
import torch.nn as nn

from torch.autograd import Variable

from models.model import PSPNet
import torch.nn.functional as F

test_file = "004810"

img_path = "data/processed/pulley_refine/" + test_file + "img.png"
img = cv2.imread(img_path)
demo = img.copy()
img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_AREA)
img = img[:, :, :3].transpose(2, 0, 1)
img = img[np.newaxis, ...]

input = Variable(torch.from_numpy(img).cuda()).float()

model = PSPNet().cuda()
model = nn.DataParallel(model)
model = torch.load("best_model_psp.pth")
model.eval()

output = model(input)

m = nn.Softmax2d()
output = m(output)
result = output[0, 1, :, :].cpu().detach().numpy()

demo = cv2.resize(demo, (800, 800), interpolation=cv2.INTER_AREA)
result_demo = cv2.resize(result, (800, 800), interpolation=cv2.INTER_AREA)

cv2.imshow("view", demo)
cv2.imshow("result", result_demo)
cv2.waitKey(0)
