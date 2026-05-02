import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward


class FreeKD(nn.Module):

    def __init__(self, teacher, student):
        super(FreeKD, self).__init__()
        self.teacher = teacher
        self.student = student

        self.conv = nn.Conv2d(256, 256, 3, 2, 1)
        self.attend = nn.Softmax(dim=-1)
        self.proj1 = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1, bias=False)
        )
        self.proj2 = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1, bias=False)
        )

        self.dwt = DWTForward(J=3, wave='db3', mode='zero')
        self.prompt = nn.Parameter(torch.zeros(1, 4, 256, 256))
        self.mse = nn.MSELoss()

    def forward(self, x):
        with torch.no_grad():
            _, (_, _, t_fea) = self.teacher(x)

        s_out, (_, _, s_fea) = self.student(x)
        s_fea = self.conv(s_fea)

        b, c, h, w = s_fea.shape
        s_a = self.attend(torch.matmul(s_fea.reshape(b, c, -1),
                                       s_fea.reshape(b, c, -1).permute(0, 2, 1)))
        t_a = self.attend(torch.matmul(t_fea.reshape(b, c, -1),
                                       t_fea.reshape(b, c, -1).permute(0, 2, 1)))

        s_w = self.proj1(s_a.permute(0, 2, 1)).unsqueeze(dim=-1)
        t_w = self.proj2(t_a.permute(0, 2, 1)).unsqueeze(dim=-1)
        weight = s_w * t_w

        prompt = self.prompt.expand(b, -1, -1, -1)
        s_l, (s_hl, s_lh, s_hh) = self.dwt(s_fea)
        t_l, (t_hl, t_lh, t_hh) = self.dwt(t_fea)

        b, c, h, w = s_l.shape
        s_ml = torch.matmul(prompt[:, 0, :, :], s_l.reshape(b, c, -1))
        s_ml = s_ml.reshape(b, c, h, w)
        t_ml = torch.matmul(prompt[:, 0, :, :], t_l.reshape(b, c, -1))
        t_ml = t_ml.reshape(b, c, h, w)

        s_hl = s_hl.mean(2)
        t_hl = t_hl.mean(2)
        b, c, h, w = s_hl.shape
        s_mhl = torch.matmul(prompt[:, 1, :, :], s_hl.reshape(b, c, -1))
        s_mhl = s_mhl.reshape(b, c, h, w)
        t_mhl = torch.matmul(prompt[:, 1, :, :], t_hl.reshape(b, c, -1))
        t_mhl = t_mhl.reshape(b, c, h, w)

        s_lh = s_lh.mean(2)
        t_lh = t_lh.mean(2)
        b, c, h, w = s_lh.shape
        s_mlh = torch.matmul(prompt[:, 2, :, :], s_lh.reshape(b, c, -1))
        s_mlh = s_mlh.reshape(b, c, h, w)
        t_mlh = torch.matmul(prompt[:, 2, :, :], t_lh.reshape(b, c, -1))
        t_mlh = t_mlh.reshape(b, c, h, w)

        s_hh = s_hh.mean(2)
        t_hh = t_hh.mean(2)
        b, c, h, w = s_hh.shape
        s_mhh = torch.matmul(prompt[:, 3, :, :], s_hh.reshape(b, c, -1))
        s_mhh = s_mhh.reshape(b, c, h, w)
        t_mhh = torch.matmul(prompt[:, 3, :, :], t_hh.reshape(b, c, -1))
        t_mhh = t_mhh.reshape(b, c, h, w)

        loss1 = self.mse(weight * s_ml, weight * t_ml)
        loss2 = self.mse(weight * s_mhl, weight * t_mhl)
        loss3 = self.mse(weight * s_mlh, weight * t_mlh)
        loss4 = self.mse(weight * s_mhh, weight * t_mhh)

        loss_freeKD = loss1 + loss2 + loss3 + loss4

        return s_out, loss_freeKD
