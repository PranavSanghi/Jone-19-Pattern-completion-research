import torch
import torch.nn as nn
from torchvision import models 

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.res = models.resnet18(weights = 'DEFAULT')
        self.res.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,1)
        )

    def forward(self,patch,img):
        img = nn.functional.interpolate(img,size=(224,224),mode='bilinear')
        patch = nn.functional.interpolate(patch,size=(224,224),mode='bilinear')
        img = self.res(img)
        patch = self.res(patch)
        x = torch.cat([img,patch],dim=1)
        
        x = self.head(x)
        return torch.sigmoid(x)

