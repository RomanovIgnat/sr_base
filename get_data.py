import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.RandomCrop((64, 64)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

screwed_transform = transforms.Compose([#transforms.Pad(padding=5),
                                        #transforms.GaussianBlur(kernel_size=(11, 11), sigma=2),
                                        transforms.Resize((32, 32), interpolation=3)])

validation_transform = transforms.Compose([transforms.CenterCrop(500),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.ImageFolder(root='/content/gdrive/MyDrive/DIV2K_train_HR', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=0)

testset = torchvision.datasets.ImageFolder(root='/content/gdrive/MyDrive/DIV2K_valid_HR', transform=validation_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
