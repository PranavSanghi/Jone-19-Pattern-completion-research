import torch
import torch.nn as nn
from torchvision import models


class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.res = models.resnet18(weights="DEFAULT")
        self.res.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, img):
        img = nn.functional.interpolate(img, size=(224, 224), mode="bilinear")
        x = self.res(img)
        x = self.head(x)
        return torch.sigmoid(x)
