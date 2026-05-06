import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def ofa_loss(logits_student, logits_teacher, target_mask, eps=1., temperature=1.):
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    prod = (pred_teacher + target_mask) ** eps
    loss = torch.sum(- (prod - target_mask) * torch.log(pred_student), dim=-1)

    return loss.mean()


class OFA(nn.Module):

    def __init__(self, teacher, student, num_classes):
        super(OFA, self).__init__()
        self.teacher = teacher
        self.student = student

        self.projector1 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, 0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes, bias=False)
        )

        self.projector2 = nn.Sequential(
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes, bias=False)
        )

        self.projector3 = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes, bias=False)
        )

    def forward(self, image, label):
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        logits_student, feat_student = self.student(image)
        num_classes = logits_student.size(-1)
        if len(label.shape) != 1:
            target_mask = F.one_hot(label.argmax(-1), num_classes)
        else:
            target_mask = F.one_hot(label, num_classes)

        ofa_loss1 = ofa_loss(self.projector1(feat_student[0]), logits_teacher, target_mask)
        ofa_loss2 = ofa_loss(self.projector2(feat_student[1]), logits_teacher, target_mask)
        ofa_loss3 = ofa_loss(self.projector3(feat_student[2]), logits_teacher, target_mask)

        loss_ofa = ofa_loss1 + ofa_loss2 + ofa_loss3

        return logits_student, loss_ofa
