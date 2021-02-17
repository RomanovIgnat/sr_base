import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


def PSNRmetrics(x, y, model):
    y_pred = model(x)
    mse = nn.MSELoss()(y_pred, y)
    return 10 * torch.log10(4.0 / mse)
