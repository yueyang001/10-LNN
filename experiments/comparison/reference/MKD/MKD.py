import torch
import torch.nn as nn
import torch.nn.functional as F

from Decoder import Decoder
from ResTeacher import resnet18
from CNNStudent import CNNStudent


def conv(inp, oup, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class MKD(nn.Module):

    def __init__(self, teacher, student, mask_ratio, depth):
        super(MKD, self).__init__()
        self.teacher = teacher
        self.student = student
        self.mask_ratio = mask_ratio

        self.conv_align1 = conv(64, 64, kernel_size=2, stride=2, padding=0)
        self.conv_align2 = conv(128, 128, kernel_size=2, stride=2, padding=0)
        self.conv_align3 = conv(256, 256, kernel_size=2, stride=2, padding=0)

        self.conv_sam1 = conv(64, 64, kernel_size=4, stride=4, padding=0)
        self.conv_sam2 = conv(128, 128, kernel_size=2, stride=2, padding=0)

        self.decoder1 = nn.Sequential(*[
            Decoder(64, (14, 14)) for _ in range(depth)
        ])
        self.decoder2 = nn.Sequential(*[
            Decoder(128, (14, 14)) for _ in range(depth)
        ])
        self.decoder3 = nn.Sequential(*[
            Decoder(256, (14, 14)) for _ in range(depth)
        ])

        self.conv_srm1 = nn.Sequential(
            conv(64, 4 * 4 * 64, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(4)
        )
        self.conv_srm2 = nn.Sequential(
            conv(128, 2 * 2 * 128, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        s_mask, t_mask = [], []

        s_out, (s_fea1, s_fea2, s_fea3) = self.student(x)
        with torch.no_grad():
            t_out, (t_fea1, t_fea2, t_fea3) = self.teacher(x)

        s_fea1 = self.conv_align1(s_fea1)
        s_fea2 = self.conv_align2(s_fea2)
        s_fea3 = self.conv_align3(s_fea3)

        s_fea1 = self.conv_sam1(s_fea1)
        s_fea2 = self.conv_sam2(s_fea2)

        s_fea1 = self.decoder1(s_fea1)
        s_fea2 = self.decoder2(s_fea2)
        s_fea3 = self.decoder3(s_fea3)

        s_fea1 = self.conv_srm1(s_fea1)
        s_fea2 = self.conv_srm2(s_fea2)

        s_mask.append(s_fea1)
        s_mask.append(s_fea2)
        s_mask.append(s_fea3)

        t_mask.append(t_fea1)
        t_mask.append(t_fea2)
        t_mask.append(t_fea3)

        return s_out, t_out, s_mask, t_mask


# if __name__ == '__main__':
#
#     x = torch.rand(16, 3, 224, 224).cuda()
#     teacher = resnet18(pretrained=True).cuda()
#     student = CNNStudent(0.1).cuda()
#     model = MMKD(teacher=teacher, student=student, mask_ratio=0.1, depth=4).cuda()
#     s_out, t_out, s_mask, t_mask = model(x)
#     print(s_out.shape, t_out.shape)
#     print(s_mask[0].shape, s_mask[1].shape, s_mask[2].shape)
#     print(t_mask[0].shape, t_mask[1].shape, t_mask[2].shape)
