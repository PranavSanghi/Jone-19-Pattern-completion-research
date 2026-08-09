import torch
import torch.nn as nn
from torchvision import models


class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.res = models.resnet50(weights="DEFAULT")
        self.res.fc = nn.Identity()
        self.head = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 1),
        )
        for param in list(self.res.parameters())[:-20]:
         param.requires_grad = False

    def forward(self, img):
        img = nn.functional.interpolate(img, size=(224, 224), mode="bilinear")
        x = self.res(img)
        x = self.head(x)
        return torch.sigmoid(x)
