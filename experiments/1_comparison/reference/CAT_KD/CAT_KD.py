import torch
import torch.nn as nn
import torch.nn.functional as F


class CAT_KD(nn.Module):

    def __init__(self, student, teacher, num_classes):
        super(CAT_KD, self).__init__()
        self.student = student
        self.teacher = teacher
        # self.conv_student = nn.Conv2d(256, num_classes, 1, 1, 0)
        # self.conv_teacher = nn.Conv2d(256, num_classes, 1, 1, 0)

    def forward(self, image):
        logit_student, feature_student = self.student(image)
        with torch.no_grad():
            logit_teacher, feature_teacher = self.teacher(image)
        tea = feature_teacher[-1]
        stu = feature_student[-1]
        s_H, t_H = stu.shape[2], tea.shape[2]
        if s_H > t_H:
            stu = F.adaptive_avg_pool2d(stu, (t_H, t_H))
        elif s_H < t_H:
            tea = F.adaptive_avg_pool2d(tea, (s_H, s_H))

        # stu = self.conv_student(stu)
        # tea = self.conv_teacher(tea)

        return logit_student, stu, tea
