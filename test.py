from models.model import FlowNet

import numpy as np
import torch


model = FlowNet()

input = torch.randn(1, 5, 240, 240).float()

output1,out2,out3 = model(input)
print(out3.shape)
