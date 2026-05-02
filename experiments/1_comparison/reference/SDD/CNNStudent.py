import torch
import torch.nn as nn
from SPP import SPP


class CNNStudent(nn.Module):

    def __init__(self):
        super().__init__()
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
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 4)
        self.spp = SPP(M='[1,2,4]')
        self.class_num = 4

    def forward(self, x):
        feat = []

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x_spp, x_strength = self.spp(x)
        x_spp = x_spp.permute((2, 0, 1))
        m, b, c = x_spp.shape[0], x_spp.shape[1], x_spp.shape[2]
        x_spp = torch.reshape(x_spp, (m * b, c))
        patch_score = self.fc(x_spp)
        patch_score = torch.reshape(patch_score, (m, b, self.class_num))
        patch_score = patch_score.permute((1, 2, 0))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, patch_score
