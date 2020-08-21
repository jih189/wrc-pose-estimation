from models.model import FlowNet

import numpy as np
import torch


model = FlowNet()

input = torch.randn(1, 5, 240, 240).float()

output, psp = model(input)
print(output.shape)
print(psp.shape)
