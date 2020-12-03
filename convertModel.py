import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import src.configuration as CFG

from models.models import Darknet  # set ONNX_EXPORT in models.py
from models.model import Magic_Net, FlowNet, DeepIM


################### magic net ########################
viewpt_class = CFG.VIEWPOINT_NUM
rot_class = CFG.ROTATION_NUM

rot_model = Magic_Net(viewpt_class=viewpt_class, rot_class=rot_class).cuda()
rot_model = torch.load(CFG.BEST_MODEL_ROT)
torch.save(rot_model.module.state_dict(), CFG.BEST_MODEL_ROT)

################# refine net ###########################
refine_model = DeepIM().cuda()
refine_model = torch.load(CFG.BEST_MODEL_ITERATIVE_REFINE)
torch.save(refine_model.module.state_dict(), CFG.BEST_MODEL_ITERATIVE_REFINE)
