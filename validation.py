import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


def PSNRmetrics(x, y, model):
    y_pred = model(x)
    mse = nn.MSELoss()((y_pred + 1) / 2, (y + 1) / 2)
    return 10 * torch.log10(1 / mse)
