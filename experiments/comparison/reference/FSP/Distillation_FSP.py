import torch
import torch.nn as nn
from FSPLoss import FSP
import torch.nn.functional as F


class DistillationModel(nn.Module):

    def __init__(self, teacher_model, student_model):
        super(DistillationModel, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.fsp = FSP()
        self.conv1 = nn.Conv2d(128, 64, 1)
        self.conv2 = nn.Conv2d(256, 128, 1)
        self.conv3 = nn.Conv2d(512, 256, 1)

    def forward(self, x):

        teacher_output, teacher_output_fsp = self.teacher_model(x)
        student_output, student_output_fsp = self.student_model(x)
        teacher_output_fsp = [self.conv1(teacher_output_fsp[0]), self.conv2(teacher_output_fsp[1]),
                              self.conv3(teacher_output_fsp[2])]
        loss1, loss2 = self.fsp(student_output_fsp, teacher_output_fsp)

        return teacher_output, student_output, loss1, loss2
