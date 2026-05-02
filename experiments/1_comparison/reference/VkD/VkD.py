import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class VkD(nn.Module):

    def __init__(self, teacher, student):
        super(VkD, self).__init__()
        self.teacher = teacher
        self.student = student
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')
        self.proj = nn.Linear(256, 256, bias=False)

    def forward(self, image):
        with torch.no_grad():
            y_t, (_, _, z_t) = self.teacher(image)

        y_s, (_, _, z_s) = self.student(image)

        b, c, h, w = z_s.shape
        z_s_pool = z_s.view(b, c, h * w).mean(-1)
        z_s_pool = self.proj(z_s_pool)
        b, c, h, w = z_t.shape
        z_t_pool = z_t.view(b, c, h * w).mean(-1)
        z_t_norm = F.layer_norm(z_t_pool, (z_t_pool.shape[1],))
        repr_distill_loss = F.smooth_l1_loss(z_s_pool, z_t_norm)

        kl_loss = self.kl(F.log_softmax(y_s / 1, dim=-1), F.softmax(y_t / 1, dim=-1)) * 1 * 1
        loss_vkd = repr_distill_loss + kl_loss

        return y_s, loss_vkd
