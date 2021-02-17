import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from main import device

validation_model = torchvision.models.vgg16(pretrained=True)
validation_model.classifier = nn.Identity()
validation_model.avgpool = nn.Identity()

for _ in range(30, 21, -1):
    validation_model.features[_] = nn.Identity()

validation_model = validation_model.to(device)


def compute_loss(x, y, model):
    y_pred = model(x)
    l1 = nn.L1Loss()(y_pred, y)
    l1_features = nn.L1Loss()(validation_model(y_pred), validation_model(y))
    return 2.5 * l1 + l1_features
