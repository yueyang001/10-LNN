import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationModel(nn.Module):

    def __init__(self, teacher_model, student_model):
        super(DistillationModel, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.conv = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):

        teacher_output, teacher_output_icc = self.teacher_model(x)
        student_output, student_output_icc = self.student_model(x)

        student_output_icc = self.conv(student_output_icc)
        batch_size, channel, _, _ = student_output_icc.shape
        student_output_icc = student_output_icc.view(batch_size, channel, -1)
        student_icc = torch.bmm(student_output_icc, student_output_icc.permute(0, 2, 1))
        teacher_output_icc = teacher_output_icc.view(batch_size, channel, -1)
        teacher_icc = torch.bmm(teacher_output_icc, teacher_output_icc.permute(0, 2, 1))

        return teacher_output, student_output, teacher_icc, student_icc
