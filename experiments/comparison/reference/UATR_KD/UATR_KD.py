import torch
import torch.nn as nn
import torch.nn.functional as F


class UATR_KD(nn.Module):

    def __init__(self, student, teacher):
        super(UATR_KD, self).__init__()
        self.student = student
        self.teacher = teacher

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

        return logit_student, logit_teacher, stu, tea
