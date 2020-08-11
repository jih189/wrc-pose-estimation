# test
import torch
import torch.nn as nn
import numpy as np
from models.model import Refine_Net

input = torch.randn(1,6,240,240).cuda()

mymodel = Refine_Net().cuda()

output = mymodel(input)