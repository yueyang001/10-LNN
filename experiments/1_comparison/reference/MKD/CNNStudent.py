import torch
import torch.nn as nn
from Mask import mask_create
import torch.nn.functional as F


class CNNStudent(nn.Module):

    def __init__(self, mask_ratio):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.upsample = nn.Upsample(scale_factor=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 4)

    def forward(self, x):
        feat = []
        B, _, _, _ = x.shape

        mask4 = mask_create(shape=(B, 1, 28, 28), mask_ratio=self.mask_ratio)
        mask3 = self.upsample(mask4)
        mask2 = self.upsample(mask3)
        mask1 = self.upsample(mask2)

        x = x * mask1
        x = self.maxpool(x)
        x = x * mask2
        x = self.block1(x)
        feat.append(x)
        x = x * mask3
        x = self.block2(x)
        feat.append(x)
        x = x * mask4
        x = self.block3(x)
        feat.append(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, feat
