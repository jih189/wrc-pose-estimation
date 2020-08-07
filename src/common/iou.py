import torch
import torch.nn as nn
import numpy as np
SMOOTH = 1e-6
def iou(labels, outputs, classindex):
    N = outputs.shape[0]
    
    pred = torch.argmax(outputs, 1, keepdim=True).squeeze(1)
    pred = (pred == classindex)
    labels = (labels == classindex)
    intersection = (pred & labels).float().sum((1,2))
    union = (pred | labels).float().sum((1,2))
    iou_v = (intersection + SMOOTH) / (union + SMOOTH)
    result = torch.clamp(20*(iou_v - 0.5), 0, 10).ceil() / 10
    result = sum(result) / N
    return result