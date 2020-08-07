# test
import torch
import torch.nn as nn
from src.common.iou import iou_pytorch
import numpy as np

output = np.array([
    [[0,0,1],
    [0,1,0],[1,0,0]]
])

output = torch.from_numpy(output)
print(output)
print(output.size())
