import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from get_data import trainloader, screwed_transform, testloader
from loss import compute_loss
from model import base_model, opt
from validation import PSNRmetrics


def imshow(img):
    img = torch.clamp((img + 1) / 2, min=0, max=1)
    npimg = img.numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    return


device = "cuda" if torch.cuda.is_available() else "cpu"

num_epochs = 200
train_loss = []
test_accuracy = []

for epoch in range(num_epochs):
    print(epoch)

    base_model.train(True)
    for i, data in enumerate(trainloader):
        y, _ = data
        x = screwed_transform(y)

        x, y = x.to(device), y.to(device)

        loss = compute_loss(x, y, base_model)
        loss.backward()
        opt.step()
        opt.zero_grad()
        train_loss.append(loss.to("cpu").data.numpy())

        if i % 10 == 0:
            print(np.mean(train_loss), end=" ")
            train_loss = []

    with torch.no_grad():
        base_model.train(False)
        for i, data in enumerate(testloader):
            y, _ = data
            h, w = y.shape[2], y.shape[3]

            c_t = torchvision.transforms.Compose([  # transforms.Pad(padding=5),
                # transforms.GaussianBlur(kernel_size=(11, 11), sigma=2),
                transforms.Resize((h // 2, w // 2))])
            x = c_t(y)

            x, y = x.to(device), y.to(device)

            acc = PSNRmetrics(x, y, base_model)
            test_accuracy.append(acc.to("cpu").data.numpy())

            if -1 == i:
                pic = base_model(x)
                x = torchvision.transforms.Resize(((h // 2) * 2, (w // 2) * 2), interpolation=4)(x)
                res = torch.cat((x, pic, y), axis=3)
                imshow(res.to("cpu").data[0])

    print(np.mean(test_accuracy))
    test_accuracy = []

