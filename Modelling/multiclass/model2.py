import torch
import torch.nn as nn
from torchvision import models

INPUT_SIZE = 672


class MultiClassMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        full_resnet = models.resnet50(weights="DEFAULT")

        self.backbone = nn.Sequential(
            full_resnet.conv1,
            full_resnet.bn1,
            full_resnet.relu,
            full_resnet.maxpool,
            full_resnet.layer1,
            full_resnet.layer2,
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, composites):
        batch_size = composites.size(0)
        num_cands = composites.size(1)

        x = composites.view(-1, 3, INPUT_SIZE, INPUT_SIZE)
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        scores = x.view(batch_size, num_cands)
        return scores
