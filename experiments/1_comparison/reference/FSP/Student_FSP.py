import torch.nn as nn


class StudentNet(nn.Module):
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
        self.fc = nn.Linear(256, 5)

    def forward(self, x):
        outputs = []
        x = self.block1(x)
        outputs.append(x)
        x = self.block2(x)
        outputs.append(x)
        x = self.block3(x)
        outputs.append(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, outputs


if __name__ == '__main__':

    import torch
    x = torch.rand(4, 3, 224, 224)
    model = StudentNet()
    output, output_list = model(x)
    print(output.shape)
    for i in output_list:
        print(i.shape)
