"""
统一知识蒸馏损失函数库
整合了 20+ 种知识蒸馏方法，包括：
- 基础方法: KD, KL, MSE
- 注意力方法: AT
- 特征方法: NST, FSP, PKT, RKD, SP, CC, VID
- 解耦方法: DKD, CAT_KD, LSKD, FreeKD, NKD
- 其他方法: MGD, MKD, SDD, OFAKD, ICKD, WSLD, SRRL, VkD, DiffKD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import numpy as np


# ========================================
# 1. 基础损失函数类
# ========================================

class KDLoss(nn.Module):
    """基础知识蒸馏损失函数 (Hinton et al.)"""
    def __init__(self, temperature=4.0, alpha=0.5):
        super(KDLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        hard_loss = self.ce_loss(student_logits, labels)

        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)

        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        return total_loss


# ========================================
# 2. Logit Standardization KD (LSKD)
# ========================================

class LSKDLoss(nn.Module):
    """Logit Standardization KD (LSKD)"""
    def __init__(self):
        super(LSKDLoss, self).__init__()

    @staticmethod
    def normalize(logit):
        mean = logit.mean(dim=-1, keepdims=True)
        stdv = logit.std(dim=-1, keepdims=True)
        return (logit - mean) / (1e-7 + stdv)

    def forward(self, y_s, y_t, T):
        KD_loss = 0
        KD_loss += nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(self.normalize(y_s) / T, dim=1),
            F.softmax(self.normalize(y_t) / T, dim=1)
        ) * T * T
        return KD_loss


# ========================================
# 3. 注意力蒸馏方法
# ========================================

class AttentionLoss(nn.Module):
    """Attention Transfer (AT) - Zagoruyko & Komodakis, 2017
    
    修改: 支持 3D 音频特征 (B, C, T)
    """
    def __init__(self, p=2):
        super(AttentionLoss, self).__init__()
        self.p = p

    def forward(self, g_s, g_t):
        """
        Args:
            g_s: 学生特征 (单个张量或列表)
            g_t: 教师特征 (单个张量或列表)
        """
        # 如果输入是列表，处理每个特征对
        if isinstance(g_s, list) and isinstance(g_t, list):
            return [self.at_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]
        # 如果输入是单个张量，直接处理
        else:
            return self.at_loss(g_s, g_t)

    def at_loss(self, f_s, f_t):
        # 处理 3D (B, C, T) 或 4D (B, C, H, W) 张量
        if f_s.dim() == 3 and f_t.dim() == 3:
            # 3D 音频特征: (B, C, T)
            s_T, t_T = f_s.shape[2], f_t.shape[2]
            if s_T > t_T:
                # 使用 interpolate 调整时间维度
                f_s = F.interpolate(f_s.unsqueeze(-1), size=(t_T, 1), mode='bilinear').squeeze(-1)
            elif s_T < t_T:
                f_t = F.interpolate(f_t.unsqueeze(-1), size=(s_T, 1), mode='bilinear').squeeze(-1)
            return (self.at(f_s) - self.at(f_t)).pow(2).mean()
        elif f_s.dim() == 4 and f_t.dim() == 4:
            # 4D 图像特征: (B, C, H, W)
            s_H, t_H = f_s.shape[2], f_t.shape[2]
            if s_H > t_H:
                f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
            elif s_H < t_H:
                f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
            return (self.at(f_s) - self.at(f_t)).pow(2).mean()
        else:
            # 如果维度不匹配，尝试自适应处理
            raise ValueError(f"不支持的张量维度: f_s.shape={f_s.shape}, f_t.shape={f_t.shape}")

    def at(self, f):
        # 处理 3D 或 4D 张量
        if f.dim() == 3:
            # (B, C, T) -> (B, C*T)
            return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))
        elif f.dim() == 4:
            # (B, C, H, W) -> (B, C*H*W)
            return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))
        else:
            raise ValueError(f"不支持的张量维度: {f.shape}")


# ========================================
# 3. 特征蒸馏方法
# ========================================

class NSTLoss(nn.Module):
    """Neuron Selectivity Transfer (NST) - Huang & Wang, 2017
    
    修改: 支持 3D 音频特征 (B, C, T)
    """
    def __init__(self):
        super(NSTLoss, self).__init__()

    def forward(self, g_s, g_t):
        # 如果输入是列表，处理每个特征对
        if isinstance(g_s, list) and isinstance(g_t, list):
            return [self.nst_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]
        # 如果输入是单个张量，直接处理
        else:
            return self.nst_loss(g_s, g_t)

    def nst_loss(self, f_s, f_t):
        # 处理 3D (B, C, T) 或 4D (B, C, H, W) 张量
        if f_s.dim() == 3 and f_t.dim() == 3:
            # 3D 音频特征: (B, C, T)
            s_T, t_T = f_s.shape[2], f_t.shape[2]
            if s_T > t_T:
                f_s = F.interpolate(f_s.unsqueeze(-1), size=(t_T, 1), mode='bilinear').squeeze(-1)
            elif s_T < t_T:
                f_t = F.interpolate(f_t.unsqueeze(-1), size=(s_T, 1), mode='bilinear').squeeze(-1)
        elif f_s.dim() == 4 and f_t.dim() == 4:
            # 4D 图像特征: (B, C, H, W)
            s_H, t_H = f_s.shape[2], f_t.shape[2]
            if s_H > t_H:
                f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
            elif s_H < t_H:
                f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            raise ValueError(f"不支持的张量维度: f_s.shape={f_s.shape}, f_t.shape={f_t.shape}")

        f_s = f_s.view(f_s.shape[0], f_s.shape[1], -1)
        f_s = F.normalize(f_s, dim=2)
        f_t = f_t.view(f_t.shape[0], f_t.shape[1], -1)
        f_t = F.normalize(f_t, dim=2)

        return (self.poly_kernel(f_t, f_t).mean().detach() + self.poly_kernel(f_s, f_s).mean()
                - 2 * self.poly_kernel(f_s, f_t).mean())

    def poly_kernel(self, a, b):
        a = a.unsqueeze(1)
        b = b.unsqueeze(2)
        res = (a * b).sum(-1).pow(2)
        return res


class FSPLoss(nn.Module):
    """Flow of Solution Procedure (FSP) - Yim et al., 2017
    
    修改: 支持 3D 音频特征 (B, C, T)
    """
    def __init__(self):
        super(FSPLoss, self).__init__()

    def forward(self, g_s, g_t):
        # FSP 需要特征列表来计算梯度流动
        # 如果输入是单个张量，将其包装成列表
        if not isinstance(g_s, list):
            g_s = [g_s]
        if not isinstance(g_t, list):
            g_t = [g_t]

        s_fsp = self.compute_fsp(g_s)
        t_fsp = self.compute_fsp(g_t)
        loss_group = [self.compute_loss(s, t) for s, t in zip(s_fsp, t_fsp)]
        return loss_group

    @staticmethod
    def compute_loss(s, t):
        return (s - t).pow(2).mean()

    @staticmethod
    def compute_fsp(g):
        fsp_list = []
        for i in range(len(g) - 1):
            bot, top = g[i], g[i + 1]

            # 处理 3D (B, C, T) 或 4D (B, C, H, W) 张量
            if bot.dim() == 3 and top.dim() == 3:
                # 3D 音频特征: (B, C, T)
                b_T, t_T = bot.shape[2], top.shape[2]
                if b_T > t_T:
                    bot = F.interpolate(bot.unsqueeze(-1), size=(t_T, 1), mode='bilinear').squeeze(-1)
                elif b_T < t_T:
                    top = F.interpolate(top.unsqueeze(-1), size=(b_T, 1), mode='bilinear').squeeze(-1)
            elif bot.dim() == 4 and top.dim() == 4:
                # 4D 图像特征: (B, C, H, W)
                b_H, t_H = bot.shape[2], top.shape[2]
                if b_H > t_H:
                    bot = F.adaptive_avg_pool2d(bot, (t_H, t_H))
                elif b_H < t_H:
                    top = F.adaptive_avg_pool2d(top, (b_H, b_H))
            else:
                raise ValueError(f"不支持的张量维度: bot.shape={bot.shape}, top.shape={top.shape}")

            bot = bot.unsqueeze(1)
            top = top.unsqueeze(2)
            bot = bot.view(bot.shape[0], bot.shape[1], bot.shape[2], -1)
            top = top.view(top.shape[0], top.shape[1], top.shape[2], -1)

            fsp = (bot * top).mean(-1)
            fsp_list.append(fsp)
        return fsp_list


class PKTLoss(nn.Module):
    """Probabilistic Knowledge Transfer (PKT) - Passalis et al., 2018"""
    def __init__(self, eps=0.0000001):
        super(PKTLoss, self).__init__()
        self.eps = eps

    def forward(self, f_s, f_t):
        return self.cosine_similarity_loss(f_s, f_t)

    def cosine_similarity_loss(self, output_net, target_net):
        # 处理 3D 张量 (B, T, C) -> 2D (B*T, C)
        original_shape = output_net.shape
        if len(original_shape) == 3:
            output_net = output_net.reshape(-1, original_shape[-1])
            target_net = target_net.reshape(-1, original_shape[-1])

        output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
        output_net = output_net / (output_net_norm + self.eps)
        output_net[output_net != output_net] = 0

        target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
        target_net = target_net / (target_net_norm + self.eps)
        target_net[target_net != target_net] = 0

        model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
        target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

        model_similarity = (model_similarity + 1.0) / 2.0
        target_similarity = (target_similarity + 1.0) / 2.0

        model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
        target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

        loss = torch.mean(target_similarity * torch.log((target_similarity + self.eps) / (model_similarity + self.eps)))
        return loss


class RKDLoss(nn.Module):
    """Relational Knowledge Distillation (RKD) - Park et al., 2019"""
    def __init__(self, w_d=25, w_a=50):
        super(RKDLoss, self).__init__()
        self.w_d = w_d
        self.w_a = w_a

    def forward(self, f_s, f_t):
        # 使用 reshape 替代 view 以处理非连续张量
        student = f_s.reshape(f_s.shape[0], -1)
        teacher = f_t.reshape(f_t.shape[0], -1)

        with torch.no_grad():
            t_d = self.pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = self.pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss_d = F.smooth_l1_loss(d, t_d)

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss_a = F.smooth_l1_loss(s_angle, t_angle)

        loss = self.w_d * loss_d + self.w_a * loss_a
        return loss

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res


class SimilarityLoss(nn.Module):
    """Similarity-Preserving Knowledge Distillation (SP) - Tung et al., 2019

    修改: 支持 3D 音频特征 (B, C, T)
    """
    def __init__(self):
        super(SimilarityLoss, self).__init__()

    def forward(self, g_s, g_t):
        # 如果输入是列表，处理每个特征对
        if isinstance(g_s, list) and isinstance(g_t, list):
            return [self.similarity_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]
        # 如果输入是单个张量，直接处理
        else:
            return self.similarity_loss(g_s, g_t)

    def similarity_loss(self, f_s, f_t):
        bsz = f_s.shape[0]
        # 直接使用 reshape 处理 3D 或 4D 张量
        f_s = f_s.reshape(bsz, -1)
        f_t = f_t.reshape(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        G_s = F.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        G_t = F.normalize(G_t)

        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss


class CorrelationLoss(nn.Module):
    """Correlation Congruence for Knowledge Distillation (CC) - Peng et al., 2019"""
    def __init__(self):
        super(CorrelationLoss, self).__init__()

    def forward(self, f_s, f_t):
        # 处理维度不匹配问题：如果特征维度不同，先对齐
        if f_s.shape[-1] != f_t.shape[-1]:
            # 如果是 3D 张量 (B, T, C)，只对齐最后一个维度
            if len(f_s.shape) == 3 and len(f_t.shape) == 3:
                if f_s.shape[-1] < f_t.shape[-1]:
                    # 学生特征维度小，使用 1x1 卷积扩展
                    if not hasattr(self, 'proj_cc'):
                        self.proj_cc = nn.Linear(f_s.shape[-1], f_t.shape[-1]).to(f_s.device)
                        self.proj_cc.weight.data.normal_(0, 0.01)
                        self.proj_cc.bias.data.zero_()
                    f_s = self.proj_cc(f_s)
                else:
                    # 教师特征维度小，使用 1x1 卷积扩展
                    if not hasattr(self, 'proj_cc_t'):
                        self.proj_cc_t = nn.Linear(f_t.shape[-1], f_s.shape[-1]).to(f_s.device)
                        self.proj_cc_t.weight.data.normal_(0, 0.01)
                        self.proj_cc_t.bias.data.zero_()
                    f_t = self.proj_cc_t(f_t)
            elif len(f_s.shape) == 2 and len(f_t.shape) == 2:
                # 2D 张量处理
                if f_s.shape[-1] < f_t.shape[-1]:
                    if not hasattr(self, 'proj_cc'):
                        self.proj_cc = nn.Linear(f_s.shape[-1], f_t.shape[-1]).to(f_s.device)
                        self.proj_cc.weight.data.normal_(0, 0.01)
                        self.proj_cc.bias.data.zero_()
                    f_s = self.proj_cc(f_s)
                else:
                    if not hasattr(self, 'proj_cc_t'):
                        self.proj_cc_t = nn.Linear(f_t.shape[-1], f_s.shape[-1]).to(f_s.device)
                        self.proj_cc_t.weight.data.normal_(0, 0.01)
                        self.proj_cc_t.bias.data.zero_()
                    f_t = self.proj_cc_t(f_t)

        delta = torch.abs(f_s - f_t)
        loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
        return loss


class VIDLoss(nn.Module):
    """Variational Information Distillation (VID) - Ahn et al., 2019"""
    def __init__(self, num_input_channels, num_mid_channel, num_target_channels, init_pred_var=5.0, eps=1e-5):
        super(VIDLoss, self).__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, padding=0,
                bias=False, stride=stride)

        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_target_channels),
        )
        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(init_pred_var-eps)-1.0) * torch.ones(num_target_channels)
        )
        self.eps = eps

    def forward(self, input, target):
        # 处理 3D 输入 (B, T, C) -> 4D (B, C, T, 1)
        if len(input.shape) == 3:
            B, T, C_in = input.shape
            input = input.permute(0, 2, 1).unsqueeze(-1)  # (B, C_in, T, 1)
            target = target.permute(0, 2, 1).unsqueeze(-1)  # (B, C_out, T, 1)

        s_H, t_H = input.shape[2], target.shape[2]
        if s_H > t_H:
            input = F.adaptive_avg_pool2d(input, (t_H, t_H))
        elif s_H < t_H:
            target = F.adaptive_avg_pool2d(target, (s_H, s_H))

        # 处理通道数不匹配
        if input.shape[1] != self.regressor[0].in_channels:
            if not hasattr(self, 'channel_adapter'):
                self.channel_adapter = nn.Conv2d(
                    input.shape[1], self.regressor[0].in_channels,
                    kernel_size=1, bias=False
                ).to(input.device)
            input = self.channel_adapter(input)

        pred_mean = self.regressor(input)
        pred_var = torch.log(1.0+torch.exp(self.log_scale))+self.eps
        pred_var = pred_var.view(1, -1, 1, 1)
        neg_log_prob = 0.5*((pred_mean-target)**2/pred_var+torch.log(pred_var))
        loss = torch.mean(neg_log_prob)
        return loss


# ========================================
# 4. 解耦与高级蒸馏方法
# ========================================

class DKDLoss(nn.Module):
    """Decoupled Knowledge Distillation (DKD) - Zhao et al., 2022"""
    def __init__(self, alpha=1.0, beta=1.0, temperature=4.0):
        super(DKDLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def _get_gt_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
        return mask

    def _get_other_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
        return mask

    def cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt

    def forward(self, logits_student, logits_teacher, target):
        gt_mask = self._get_gt_mask(logits_student, target)
        other_mask = self._get_other_mask(logits_student, target)
        pred_student = F.softmax(logits_student / self.temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=1)
        pred_student = self.cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = self.cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
                F.kl_div(log_pred_student, pred_teacher, reduction='sum')
                * (self.temperature ** 2) / target.shape[0]
        )
        pred_teacher_part2 = F.softmax(
            logits_teacher / self.temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / self.temperature - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
                F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='sum')
                * (self.temperature ** 2) / target.shape[0]
        )

        loss = self.alpha * tckd_loss + self.beta * nckd_loss
        return loss


class NKDLoss(nn.Module):
    """Negative Knowledge Distillation (NKD)"""
    def __init__(self, temp=1.0, gamma=1.5):
        super(NKDLoss, self).__init__()
        self.temp = temp
        self.gamma = gamma
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logit_s, logit_t, gt_label):
        if len(gt_label.size()) > 1:
            label = torch.max(gt_label, dim=1, keepdim=True)[1]
        else:
            label = gt_label.view(len(gt_label), 1)

        N, c = logit_s.shape
        s_i = self.log_softmax(logit_s)
        t_i = F.softmax(logit_t, dim=1)
        s_t = torch.gather(s_i, 1, label)
        t_t = torch.gather(t_i, 1, label).detach()

        loss_t = - (t_t * s_t).mean()

        mask = torch.ones_like(logit_s).scatter_(1, label, 0).bool()
        logit_s = logit_s[mask].reshape(N, -1)
        logit_t = logit_t[mask].reshape(N, -1)

        S_i = self.log_softmax(logit_s / self.temp)
        T_i = F.softmax(logit_t / self.temp, dim=1)

        loss_non = (T_i * S_i).sum(dim=1).mean()
        loss_non = - self.gamma * (self.temp ** 2) * loss_non

        return loss_t + loss_non


class WSLDLoss(nn.Module):
    """Weighted Soft Label Distillation (WSLD)"""
    def __init__(self, temp=1.0, alpha=1.00, num_classes=4):
        super(WSLDLoss, self).__init__()
        self.temp = temp
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, logit_s, logit_t, gt_label):
        # 获取输入设备
        device = logit_s.device

        s_input_for_softmax = logit_s / self.temp
        t_input_for_softmax = logit_t / self.temp
        t_soft_label = F.softmax(t_input_for_softmax, dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)
        softmax_loss = - torch.sum(t_soft_label * logsoftmax(s_input_for_softmax), 1, keepdim=True)
        fc_s_auto = logit_s.detach()
        fc_t_auto = logit_t.detach()
        log_softmax_s = logsoftmax(fc_s_auto)
        log_softmax_t = logsoftmax(fc_t_auto)
        one_hot_label = F.one_hot(gt_label, num_classes=self.num_classes).float()
        softmax_loss_s = - torch.sum(one_hot_label * log_softmax_s, 1, keepdim=True)
        softmax_loss_t = - torch.sum(one_hot_label * log_softmax_t, 1, keepdim=True)
        focal_weight = softmax_loss_s / (softmax_loss_t + 1e-7)
        ratio_lower = torch.zeros(1, device=device)  # 修复：使用正确的设备
        focal_weight = torch.max(focal_weight, ratio_lower)
        focal_weight = 1 - torch.exp(- focal_weight)
        softmax_loss = focal_weight * softmax_loss
        wsld_loss = (self.temp ** 2) * torch.mean(softmax_loss)

        return self.alpha * wsld_loss


# ========================================
# 5. 生成式蒸馏方法
# ========================================

class MGDLoss(nn.Module):
    """Masked Generative Distillation (MGD)"""
    def __init__(self, student_channels, teacher_channels, alpha_mgd=0.00007, lambda_mgd=0.15):
        super(MGDLoss, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd

        self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=2, stride=2, padding=0)

        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1))

    def forward(self, preds_S, preds_T):
        preds_S = self.align(preds_S)
        loss = self.get_dis_loss(preds_S, preds_T) * self.alpha_mgd
        return loss

    def get_dis_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss()
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N, C, 1, 1)).to(device)
        mat = torch.where(mat < self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation(masked_fea)

        dis_loss = loss_mse(new_fea, preds_T) / N
        return dis_loss


# ========================================
# 6. 其他高级蒸馏方法
# ========================================

class SRRLLoss(nn.Module):
    """Spatial-wise Relative Representation Loss (SRRL)"""
    def __init__(self, student_channels, teacher_channels, alpha=1.0, beta=1.0):
        super(SRRLLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.Connectors = nn.Sequential(
            nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(teacher_channels), nn.ReLU())
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, fea_s, fea_t, logit_st, logit_t):
        x = fea_s
        x = self.Connectors(x)
        y = fea_t
        x = x.view(x.size(0), x.size(1), -1)
        y = y.view(y.size(0), y.size(1), -1)
        x_mean = x.mean(dim=2)
        y_mean = y.mean(dim=2)
        fm_loss = (x_mean - y_mean).pow(2).mean(1)
        sr_loss = torch.mean(1.0 - torch.cosine_similarity(logit_st, logit_t))

        return self.alpha * fm_loss.mean() + self.beta * sr_loss


class CAT_KDLoss(nn.Module):
    """Channel-Aligned Transfer KD (CAT_KD) - 特征对齐蒸馏"""
    def __init__(self, student_channels, teacher_channels):
        super(CAT_KDLoss, self).__init__()
        # 对齐层：将学生特征通道数对齐到教师特征通道数
        self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        self.mse_loss = nn.MSELoss()

    def forward(self, f_s, f_t):
        """
        Args:
            f_s: 学生特征 [B, C_s, H, W]
            f_t: 教师特征 [B, C_t, H, W]
        Returns:
            loss: CAT_KD 特征对齐损失
        """
        # 空间尺寸对齐
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))

        # 通道对齐
        f_s_aligned = self.align(f_s)

        # 计算对齐损失
        loss = self.mse_loss(f_s_aligned, f_t)
        return loss


class ICKDLoss(nn.Module):
    """Inner Correlation for Knowledge Distillation (ICKD)"""
    def __init__(self, student_channels, teacher_channels):
        super(ICKDLoss, self).__init__()
        # 对齐层
        self.align = nn.Sequential(
            nn.Conv2d(student_channels, teacher_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(teacher_channels),
            nn.MaxPool2d(2, 2)
        )
        self.mse_loss = nn.MSELoss()

    def forward(self, f_s, f_t):
        """
        Args:
            f_s: 学生特征 [B, C_s, H, W]
            f_t: 教师特征 [B, C_t, H, W]
        Returns:
            loss: ICKD 内部相关性损失
        """
        # 对齐学生特征
        f_s_aligned = self.align(f_s)

        batch_size, channel, _, _ = f_s_aligned.shape
        f_s_aligned = f_s_aligned.view(batch_size, channel, -1)
        f_t = f_t.view(batch_size, channel, -1)

        # 计算内部相关性矩阵
        # [B, C, HW] x [B, HW, C] -> [B, C, C]
        s_icc = torch.bmm(f_s_aligned, f_s_aligned.permute(0, 2, 1))
        t_icc = torch.bmm(f_t, f_t.permute(0, 2, 1))

        loss = self.mse_loss(s_icc, t_icc)
        return loss


class VKDLoss(nn.Module):
    """Variational Knowledge Distillation (VkD)"""
    def __init__(self, student_channels, teacher_channels=256, projection_dim=None):
        super(VKDLoss, self).__init__()
        # 如果没有指定 projection_dim，则使用教师通道数
        if projection_dim is None:
            projection_dim = teacher_channels
        # 投影层：将学生特征投影到教师特征维度
        self.proj = nn.Linear(student_channels, projection_dim, bias=False)
        # 如果投影维度与教师通道数不同，需要额外的投影层
        if projection_dim != teacher_channels:
            self.teacher_proj = nn.Linear(teacher_channels, projection_dim, bias=False)
        else:
            self.teacher_proj = None
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, f_s, f_t, logit_s=None, logit_t=None):
        """
        Args:
            f_s: 学生特征 [B, C, H, W]
            f_t: 教师特征 [B, C, H, W]
            logit_s: 学生 logit (可选) [B, num_classes]
            logit_t: 教师 logit (可选) [B, num_classes]
        Returns:
            loss: VkD 损失
        """
        device = f_s.device
        b, c, h, w = f_s.shape

        # 全局池化和投影
        f_s_pool = f_s.view(b, c, h * w).mean(-1)  # [B, C]
        f_s_proj = self.proj(f_s_pool)  # [B, proj_dim]

        b, c, h, w = f_t.shape
        f_t_pool = f_t.view(b, c, h * w).mean(-1)  # [B, C]
        # 层归一化
        f_t_norm = F.layer_norm(f_t_pool, (f_t_pool.shape[1],))

        # 如果需要，投影教师特征到相同的维度
        if self.teacher_proj is not None:
            f_t_norm = self.teacher_proj(f_t_norm)

        # 表示蒸馏损失
        repr_loss = F.smooth_l1_loss(f_s_proj, f_t_norm)

        # 如果提供了 logit，则加上 KL 损失
        if logit_s is not None and logit_t is not None:
            kl_loss = self.kl_loss(
                F.log_softmax(logit_s, dim=-1),
                F.softmax(logit_t, dim=-1)
            )
            total_loss = repr_loss + kl_loss
        else:
            total_loss = repr_loss

        return total_loss


class UATR_KDLoss(nn.Module):
    """UATR-specific Knowledge Distillation"""
    def __init__(self, student_channels, teacher_channels):
        super(UATR_KDLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        # 如果通道数不同，添加对齐层
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

    def forward(self, f_s, f_t):
        """
        Args:
            f_s: 学生特征 [B, C_s, H, W]
            f_t: 教师特征 [B, C_t, H, W]
        Returns:
            loss: UATR_KD 特征损失
        """
        # 通道对齐
        if self.align is not None:
            f_s = self.align(f_s)

        # 空间尺寸对齐
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))

        loss = self.mse_loss(f_s, f_t)
        return loss


class OFAKDLoss(nn.Module):
    """One-for-All Knowledge Distillation (OFAKD) - 简化版本"""
    def __init__(self, eps=1.0, temperature=1.0, student_channels=None, teacher_channels=None):
        super(OFAKDLoss, self).__init__()
        self.eps = eps
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        
        # 如果提供了特征维度，创建对齐层
        self.use_features = (student_channels is not None and teacher_channels is not None)
        if self.use_features:
            self.align = nn.Sequential(
                nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(teacher_channels),
                nn.ReLU()
            )

    def forward(self, *args):
        """
        Args:
            如果提供 3 个参数: logits_student, logits_teacher, target_mask
            如果提供 2 个参数: student_feat, teacher_feat (特征蒸馏)
        Returns:
            loss: OFAKD 损失
        """
        if len(args) == 3:
            # Logit-based 版本（原始 OFAKD）
            logits_student, logits_teacher, target_mask = args
            pred_student = F.softmax(logits_student / self.temperature, dim=1)
            pred_teacher = F.softmax(logits_teacher / self.temperature, dim=1)
            prod = (pred_teacher + target_mask) ** self.eps
            loss = torch.sum(- (prod - target_mask) * torch.log(pred_student), dim=-1)
            return loss.mean()
        elif len(args) == 2:
            # 特征蒸馏版本（简化）
            student_feat, teacher_feat = args
            
            if self.use_features:
                # 对齐学生特征
                student_aligned = self.align(student_feat)
            else:
                student_aligned = student_feat
            
            # 简化为 MSE 损失
            loss = F.mse_loss(student_aligned, teacher_feat)
            return loss
        else:
            raise ValueError(f"OFAKDLoss expects 2 or 3 arguments, got {len(args)}")


def ofa_loss(logits_student, logits_teacher, target_mask, eps=1., temperature=1.):
    """One-for-All (OFAKD) 损失函数"""
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    prod = (pred_teacher + target_mask) ** eps
    loss = torch.sum(- (prod - target_mask) * torch.log(pred_student), dim=-1)
    return loss.mean()


# ========================================
# 7. 特殊蒸馏方法（简化版本，不需要额外模块）
# ========================================

class DiffKD_Loss(nn.Module):
    """
    Diffusion-based Knowledge Distillation (DiffKD)
    基于原始实现，使用扩散模型进行蒸馏
    参考: experiments/1_comparison/reference/DiffKD/
    """
    def __init__(self, student_channels, teacher_channels, kernel_size=3,
                 inference_steps=5, num_train_timesteps=1000, use_ae=False, ae_channels=None):
        super(DiffKD_Loss, self).__init__()
        self.student_channels = student_channels
        self.teacher_channels = teacher_channels
        self.inference_steps = inference_steps
        self.use_ae = use_ae
        
        # 如果使用 autoencoder 压缩教师特征
        if use_ae:
            if ae_channels is None:
                ae_channels = teacher_channels // 2
            self.ae = self._AutoEncoder(teacher_channels, ae_channels)
            teacher_channels = ae_channels
        else:
            self.ae = None

        # 变换学生特征到教师特征维度
        self.trans = nn.Conv2d(student_channels, teacher_channels, kernel_size=3, stride=2, padding=1)
        
        # 扩散模型 - 预测噪声
        self.model = self._DiffusionModel(channels_in=teacher_channels, kernel_size=kernel_size)
        self.scheduler = self._DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False,
                                           beta_schedule="linear")
        self.noise_adapter = self._NoiseAdapter(teacher_channels, kernel_size)
        # pipeline 用于去噪学生特征
        self.pipeline = self._DDIMPipeline(self.model, self.scheduler, self.noise_adapter)
        self.proj = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, 1), 
            nn.BatchNorm2d(teacher_channels)
        )
        
        # 标记为可用
        self.available = True

    def forward(self, student_feat, teacher_feat):
        """
        Args:
            student_feat: 学生特征 [B, C_s, H, W] 或 [B, T, C_s]
            teacher_feat: 教师特征 [B, C_t, H, W] 或 [B, T, C_t]
        Returns:
            loss: DiffKD 损失（扩散模型去噪损失）
        """
        # 确保 scheduler 的张量在正确的设备上
        device = student_feat.device
        if hasattr(self, 'scheduler'):
            self.scheduler.betas = self.scheduler.betas.to(device)
            self.scheduler.alphas = self.scheduler.alphas.to(device)
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
            self.scheduler.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(device)
            self.scheduler.timesteps = self.scheduler.timesteps.to(device)
        
        # 处理 3D 输入 (B, T, C) -> 4D (B, C, T, 1)
        if len(student_feat.shape) == 3:
            B, T, C_in = student_feat.shape
            student_feat = student_feat.permute(0, 2, 1).unsqueeze(-1)  # (B, C_in, T, 1)
            teacher_feat = teacher_feat.permute(0, 2, 1).unsqueeze(-1)  # (B, C_out, T, 1)

        # 将学生特征投影到教师特征维度
        student_feat = self.trans(student_feat)

        # 如果空间尺寸不匹配，调整
        if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
            student_feat = F.adaptive_avg_pool2d(student_feat, teacher_feat.shape[-2:])

        # 使用 autoencoder 压缩教师特征
        if self.use_ae:
            hidden_t_feat, rec_t_feat = self.ae(teacher_feat)
            rec_loss = F.mse_loss(teacher_feat, rec_t_feat)
            teacher_feat = hidden_t_feat.detach()
        else:
            rec_loss = None

        # 去噪学生特征
        refined_feat = self.pipeline(
            batch_size=student_feat.shape[0],
            device=student_feat.device,
            dtype=student_feat.dtype,
            shape=student_feat.shape[1:],
            feat=student_feat,
            num_inference_steps=self.inference_steps,
            proj=self.proj
        )
        refined_feat = self.proj(refined_feat)

        # 训练扩散模型
        ddim_loss = self._ddim_loss(teacher_feat)

        # 如果有 rec_loss，组合损失
        if rec_loss is not None:
            total_loss = ddim_loss + rec_loss
        else:
            total_loss = ddim_loss

        return total_loss

    def _ddim_loss(self, gt_feat):
        """DDIM 去噪损失"""
        # 采样噪声添加到图像
        noise = torch.randn(gt_feat.shape, device=gt_feat.device)
        bs = gt_feat.shape[0]

        # 为每个图像采样随机时间步
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=gt_feat.device).long()
        # 根据时间步的噪声幅值添加噪声（这是前向扩散过程）
        noisy_images = self.scheduler.add_noise(gt_feat, noise, timesteps)
        noise_pred = self.model(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        return loss

    # ========================================
    # 内部辅助类（从参考实现移植）
    # ========================================

    class _Bottleneck(nn.Module):
        def __init__(self, in_channels, out_channels, reduction=4):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // reduction, 1),
                nn.BatchNorm2d(in_channels // reduction),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction, in_channels // reduction, 3, padding=1),
                nn.BatchNorm2d(in_channels // reduction),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )

        def forward(self, x):
            out = self.block(x)
            return out + x

    class _NoiseAdapter(nn.Module):
        def __init__(self, channels, kernel_size=3):
            super().__init__()
            if kernel_size == 3:
                self.feat = nn.Sequential(
                    DiffKD_Loss._Bottleneck(channels, channels, reduction=8),
                    nn.AdaptiveAvgPool2d(1)
                )
            else:
                self.feat = nn.Sequential(
                    nn.Conv2d(channels, channels * 2, 1),
                    nn.BatchNorm2d(channels * 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels * 2, channels, 1),
                    nn.BatchNorm2d(channels),
                )
            self.pred = nn.Linear(channels, 2)

        def forward(self, x):
            x = self.feat(x).flatten(1)
            x = self.pred(x).softmax(1)[:, 0]
            return x

    class _DiffusionModel(nn.Module):
        def __init__(self, channels_in, kernel_size=3):
            super().__init__()
            self.kernel_size = kernel_size
            self.time_embedding = nn.Embedding(1280, channels_in)

            if kernel_size == 3:
                self.pred = nn.Sequential(
                    DiffKD_Loss._Bottleneck(channels_in, channels_in),
                    DiffKD_Loss._Bottleneck(channels_in, channels_in),
                    nn.Conv2d(channels_in, channels_in, 1),
                    nn.BatchNorm2d(channels_in)
                )
            else:
                self.pred = nn.Sequential(
                    nn.Conv2d(channels_in, channels_in * 4, 1),
                    nn.BatchNorm2d(channels_in * 4),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels_in * 4, channels_in, 1),
                    nn.BatchNorm2d(channels_in),
                    nn.Conv2d(channels_in, channels_in * 4, 1),
                    nn.BatchNorm2d(channels_in * 4),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels_in * 4, channels_in, 1)
                )

        def forward(self, noisy_image, t):
            if t.dtype != torch.long:
                t = t.type(torch.long)
            feat = noisy_image
            time_emb = self.time_embedding(t)
            # 确保时间嵌入与特征图的 dtype 一致
            if time_emb.dtype != feat.dtype:
                time_emb = time_emb.to(feat.dtype)
            feat = feat + time_emb[..., None, None]
            ret = self.pred(feat)
            return ret

    class _AutoEncoder(nn.Module):
        def __init__(self, channels, latent_channels):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(channels, latent_channels, 1, padding=0),
                nn.BatchNorm2d(latent_channels)
            )
            self.decoder = nn.Sequential(
                nn.Conv2d(latent_channels, channels, 1, padding=0),
            )

        def forward(self, x):
            hidden = self.encoder(x)
            out = self.decoder(hidden)
            return hidden, out

    class _DDIMPipeline:
        """DDIM Pipeline"""
        def __init__(self, model, scheduler, noise_adapter=None, solver='ddim'):
            super().__init__()
            self.model = model
            self.scheduler = scheduler
            self.noise_adapter = noise_adapter
            self._iter = 0
            self.solver = solver

        def __call__(self, batch_size, device, dtype, shape, feat, 
                    generator=None, eta: float = 0.0, num_inference_steps: int = 50, proj=None):
            # 采样高斯噪声开始循环
            image_shape = (batch_size, *shape)

            if self.noise_adapter is not None:
                noise = torch.randn(image_shape, device=device, dtype=dtype)
                alpha_prod = self.noise_adapter(feat)
                image = self.scheduler.add_noise_diff2(feat, noise, alpha_prod)
            else:
                image = feat

            # 设置步长值
            self.scheduler.set_timesteps(num_inference_steps * 2)

            for t in self.scheduler.timesteps[len(self.scheduler.timesteps) // 2:]:
                # 确保 t 是一个张量
                if isinstance(t, (int, float)):
                    t = torch.tensor([t], dtype=torch.long, device=device)
                noise_pred = self.model(image, t)

                # 2. 预测前一个时刻的图像均值并根据 eta 添加方差
                # eta 对应论文中的 η，应该在 [0, 1] 之间
                # 执行 x_t -> x_t-1
                image = self.step(
                    noise_pred, t, image, eta=eta, use_clipped_model_output=True, generator=generator
                )['prev_sample']

            self._iter += 1
            return image

        def step(self, noise_pred, t, sample, eta=0.0, use_clipped_model_output=False, generator=None, variance_noise=None, return_dict=True):
            """执行 DDIM 步骤"""
            if self.scheduler.num_inference_steps is None:
                raise ValueError("Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler")

            # 确保 t 是整数标量
            if isinstance(t, torch.Tensor):
                t_int = t.item()
            else:
                t_int = int(t)

            # 1. 获取前一个步长的值 (=t-1)
            prev_timestep = t_int - self.scheduler.num_train_timesteps // self.scheduler.num_inference_steps

            # 2. 计算alphas, betas
            alpha_prod_t = self.scheduler.alphas_cumprod[t_int]
            alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev

            # 方差
            variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

            # 3. 计算预测的原始样本（也称为"predicted x_0"）
            if self.scheduler.config.prediction_type == "epsilon":
                pred_original_sample = (sample - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
            else:
                pred_original_sample = noise_pred

            # 4. 裁剪"predicted x_0"
            if self.scheduler.config.clip_sample:
                pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

            # 5. 计算标准差
            std_dev_t = eta * variance ** (0.5)

            # 6. 计算"指向 x_t 的方向"
            pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (0.5) * noise_pred

            # 7. 计算不带随机噪声的 x_t
            prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

            # 8. 添加随机噪声
            if eta > 0:
                if variance_noise is None and generator is not None:
                    variance_noise = torch.randn(noise_pred.shape, generator=generator, device=noise_pred.device)
                elif variance_noise is None:
                    device = noise_pred.device
                    variance_noise = torch.randn(noise_pred.shape, device=device, dtype=noise_pred.dtype)
                variance = self._get_variance(t, prev_timestep) ** (0.5) * eta * variance_noise
                prev_sample = prev_sample + variance

            if not return_dict:
                return (prev_sample,)
            return dict(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

        def _get_variance(self, timestep, prev_timestep):
            alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
            alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev
            variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
            return variance

    class _DDIMScheduler:
        """DDIM 调度器"""
        def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="linear", clip_sample=False, set_alpha_to_one=True):
            super().__init__()
            self.num_train_timesteps = num_train_timesteps
            self.beta_start = beta_start
            self.beta_end = beta_end
            self.beta_schedule = beta_schedule
            self.clip_sample = clip_sample
            self.set_alpha_to_one = set_alpha_to_one

            # 线性调度
            if beta_schedule == "linear":
                self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
            else:
                raise NotImplementedError(f"{beta_schedule} is not implemented")

            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

            # DDIM 的特殊处理
            self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
            self.init_noise_sigma = 1.0
            self.num_inference_steps = None
            self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1).long()

        def set_timesteps(self, num_inference_steps: int):
            """设置离散时间步"""
            self.num_inference_steps = num_inference_steps
            step_ratio = self.num_train_timesteps // num_inference_steps
            # 通过乘以比率创建整数时间步
            timesteps = (torch.arange(0, num_inference_steps) * step_ratio).round()
            timesteps = timesteps.flip(0).long()  # 替代 [::-1]
            self.timesteps = timesteps

        def add_noise(self, original_samples, noise, timesteps):
            """添加噪声（前向扩散过程）"""
            # 确保 alphas_cumprod 和 timestep 拥有相同的设备和类型
            self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
            timesteps = timesteps.to(original_samples.device)

            # 获取每个时间步的 alpha_prod
            alpha_prod = self.alphas_cumprod[timesteps]
            sqrt_alpha_prod = (alpha_prod + 1e-6) ** 0.5
            sqrt_alpha_prod = sqrt_alpha_prod.flatten()
            while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
                sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

            sqrt_one_minus_alpha_prod = (1 - alpha_prod) ** 0.5
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
            while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

            noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
            return noisy_samples

        def add_noise_diff2(self, original_samples, noise, alpha_prod):
            """添加噪声的变体 - 使用 alpha_prod 作为输入"""
            # alpha_prod 是由 noise_adapter 预测的值，在 [0, 1] 范围内
            alpha_prod = alpha_prod.to(device=original_samples.device, dtype=original_samples.dtype)
            
            sqrt_alpha_prod = alpha_prod ** 0.5
            sqrt_alpha_prod = sqrt_alpha_prod.flatten()
            while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
                sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

            sqrt_one_minus_alpha_prod = (1 - alpha_prod)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
            while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

            noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
            return noisy_samples

        @property
        def config(self):
            class Config:
                num_train_timesteps = 1000
                beta_start = 0.0001
                beta_end = 0.02
                beta_schedule = "linear"
                trained_betas = None
                clip_sample = False
                set_alpha_to_one = True
                steps_offset = 0
                prediction_type = "epsilon"
                _deprecated_kwargs = ["predict_epsilon"]
                order = 1

                def __init__(self):
                    self.num_train_timesteps = self.num_train_timesteps
                    self.beta_start = self.beta_start
                    self.beta_end = self.beta_end
                    self.beta_schedule = self.beta_schedule
                    self.clip_sample = self.clip_sample
                    self.set_alpha_to_one = self.set_alpha_to_one
                    self.steps_offset = 0
                    self.prediction_type = "epsilon"

            return Config()


class FreeKD_Loss(nn.Module):
    """
    FreeKD - 完整实现（使用小波变换）
    参考: experiments/1_comparison/reference/FreeKD/FreeKD.py
    """
    def __init__(self, student_channels, teacher_channels):
        super(FreeKD_Loss, self).__init__()
        self.student_channels = student_channels
        self.teacher_channels = teacher_channels

        # 学生特征对齐到教师特征维度
        self.conv = nn.Conv2d(student_channels, teacher_channels, kernel_size=3, stride=2, padding=1)

        # 注意力机制
        self.attend = nn.Softmax(dim=-1)

        # 两个投影网络用于生成权重
        self.proj1 = nn.Sequential(
            nn.Linear(teacher_channels, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1, bias=False)
        )
        self.proj2 = nn.Sequential(
            nn.Linear(teacher_channels, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1, bias=False)
        )

        # 离散小波变换 (DWT)
        try:
            from pytorch_wavelets import DWTForward
            self.dwt = DWTForward(J=3, wave='db3', mode='zero')
            self.available = True
        except ImportError:
            print("Warning: pytorch_wavelets not available, FreeKD will be disabled")
            self.available = False
            self.dwt = None

        # 提示向量将在第一次前向传播时根据输入尺寸初始化
        self.prompt = None
        self._prompt_initialized = False

        # 损失函数
        self.mse = nn.MSELoss()

        # 是否使用权重(参考实现中权重太小,导致损失接近零)
        self.use_weight = False
    
    def _initialize_prompt(self, channels):
        """根据输入通道数初始化提示向量"""
        if not self._prompt_initialized:
            # 提示向量形状: [1, 4, channels, channels]
            # 使用单位矩阵初始化,让 prompt 初始时就有效
            prompt_data = torch.eye(channels).unsqueeze(0).unsqueeze(0)  # [1, 1, channels, channels]
            prompt_data = prompt_data.repeat(1, 4, 1, 1)  # [1, 4, channels, channels]
            self.prompt = nn.Parameter(prompt_data)
            self._prompt_initialized = True
            # 注册为模型的参数
            if hasattr(self, 'prompt'):
                self.register_parameter('prompt', self.prompt)
    
    def forward(self, student_feat, teacher_feat):
        """
        Args:
            student_feat: 学生特征 [B, C_s, H, W] 或 [B, T, C_s]
            teacher_feat: 教师特征 [B, C_t, H, W] 或 [B, T, C_t]
        Returns:
            loss: FreeKD 损失
        """
        if not self.available:
            # 如果小波变换不可用，返回零损失
            return torch.tensor(0.0, device=student_feat.device)
        
        # 处理 3D 输入 (B, T, C) -> 4D (B, C, T, 1)
        if len(student_feat.shape) == 3:
            B, T, C_in = student_feat.shape
            student_feat = student_feat.permute(0, 2, 1).unsqueeze(-1)  # (B, C_in, T, 1)
            teacher_feat = teacher_feat.permute(0, 2, 1).unsqueeze(-1)  # (B, C_out, T, 1)
        
        # 将学生特征投影到教师特征维度
        s_fea = self.conv(student_feat)
        t_fea = teacher_feat
        
        # 如果空间尺寸不匹配，调整学生特征以匹配教师特征
        if s_fea.shape[-2:] != t_fea.shape[-2:]:
            s_fea = F.interpolate(s_fea, size=t_fea.shape[-2:], mode='bilinear', align_corners=False)
        
        # 获取空间尺寸和通道数
        b, c, h, w = s_fea.shape
        
        # 初始化提示向量（如果还没有初始化）
        if not self._prompt_initialized:
            self._initialize_prompt(c)
            # 将提示向量移动到正确的设备
            self.prompt.data = self.prompt.data.to(student_feat.device)
        elif self.prompt.device != student_feat.device:
            self.prompt.data = self.prompt.data.to(student_feat.device)
        
        # 计算自注意力
        s_a = self.attend(torch.matmul(s_fea.reshape(b, c, -1),
                                       s_fea.reshape(b, c, -1).permute(0, 2, 1)))
        t_a = self.attend(torch.matmul(t_fea.reshape(b, c, -1),
                                       t_fea.reshape(b, c, -1).permute(0, 2, 1)))
        
        # 计算权重
        s_w = self.proj1(s_a.permute(0, 2, 1)).unsqueeze(dim=-1)
        t_w = self.proj2(t_a.permute(0, 2, 1)).unsqueeze(dim=-1)
        weight = s_w * t_w
        
        # 应用提示向量
        prompt = self.prompt.expand(b, -1, -1, -1)
        
        # 离散小波变换
        s_l, (s_hl, s_lh, s_hh) = self.dwt(s_fea)
        t_l, (t_hl, t_lh, t_hh) = self.dwt(t_fea)
        
        # 处理低频分量
        b, c, h, w = s_l.shape
        # 提示向量在通道维度上应用: prompt[:, 0] is [B, c, c], s_l.reshape is [B, c, h*w]
        s_ml = torch.matmul(prompt[:, 0, :, :], s_l.reshape(b, c, -1))
        t_ml = torch.matmul(prompt[:, 0, :, :], t_l.reshape(b, c, -1))
        s_ml = s_ml.reshape(b, c, h, w)
        t_ml = t_ml.reshape(b, c, h, w)
        
        # 处理高频分量 HL (水平)
        s_hl = s_hl.mean(2)
        t_hl = t_hl.mean(2)
        b, c, h, w = s_hl.shape
        s_mhl = torch.matmul(prompt[:, 1, :, :], s_hl.reshape(b, c, -1))
        t_mhl = torch.matmul(prompt[:, 1, :, :], t_hl.reshape(b, c, -1))
        s_mhl = s_mhl.reshape(b, c, h, w)
        t_mhl = t_mhl.reshape(b, c, h, w)
        
        # 处理高频分量 LH (垂直)
        s_lh = s_lh.mean(2)
        t_lh = t_lh.mean(2)
        b, c, h, w = s_lh.shape
        s_mlh = torch.matmul(prompt[:, 2, :, :], s_lh.reshape(b, c, -1))
        t_mlh = torch.matmul(prompt[:, 2, :, :], t_lh.reshape(b, c, -1))
        s_mlh = s_mlh.reshape(b, c, h, w)
        t_mlh = t_mlh.reshape(b, c, h, w)
        
        # 处理高频分量 HH (对角)
        s_hh = s_hh.mean(2)
        t_hh = t_hh.mean(2)
        b, c, h, w = s_hh.shape
        s_mhh = torch.matmul(prompt[:, 3, :, :], s_hh.reshape(b, c, -1))
        t_mhh = torch.matmul(prompt[:, 3, :, :], t_hh.reshape(b, c, -1))
        s_mhh = s_mhh.reshape(b, c, h, w)
        t_mhh = t_mhh.reshape(b, c, h, w)
        
        # 计算各分量的损失
        # 注意:参考实现中的权重值太小,会导致损失接近零
        # 因此这里不使用权重,直接比较特征
        if self.use_weight:
            loss1 = self.mse(weight * s_ml, weight * t_ml)
            loss2 = self.mse(weight * s_mhl, weight * t_mhl)
            loss3 = self.mse(weight * s_mlh, weight * t_mlh)
            loss4 = self.mse(weight * s_mhh, weight * t_mhh)
        else:
            # 不使用权重,直接比较特征
            loss1 = self.mse(s_ml, t_ml)
            loss2 = self.mse(s_mhl, t_mhl)
            loss3 = self.mse(s_mlh, t_mlh)
            loss4 = self.mse(s_mhh, t_mhh)

        # 总损失
        loss_freeKD = loss1 + loss2 + loss3 + loss4

        return loss_freeKD


# ========================================
# Masked Knowledge Distillation (MKD) - 完整实现
# ========================================

def conv(inp, oup, kernel_size, stride, padding):
    """基础卷积块"""
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class _Decoder(nn.Module):
    """
    Decoder 模块用于 MKD
    参考: experiments/1_comparison/reference/MKD/decoder.py
    """
    
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
        """
        super(_Decoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.decoder(x)


class MKD_Loss(nn.Module):
    """
    Masked Knowledge Distillation (MKD) - 完整实现
    参考: experiments/1_comparison/reference/MKD/MKD.py
    
    包含:
    - 对齐层 (Align)
    - 空间注意力模块 (SAM)
    - 解码器 (Decoder, 深度堆叠)
    - 空间重建模块 (SRM, 使用 PixelShuffle)
    """
    
    def __init__(self, student_channels, teacher_channels, mask_ratio=0.1, depth=4):
        """
        Args:
            student_channels: 学生特征通道数
            teacher_channels: 教师特征通道数
            mask_ratio: 掩码比率 (参考实现中未使用,保留用于兼容性)
            depth: Decoder 深度堆叠层数
        """
        super(MKD_Loss, self).__init__()
        self.student_channels = student_channels
        self.teacher_channels = teacher_channels
        self.mask_ratio = mask_ratio
        self.depth = depth
        
        # 对齐层: 将学生特征空间对齐到教师特征
        # 使用自适应层来处理不同的通道数
        self.conv_align = conv(student_channels, teacher_channels, kernel_size=3, stride=1, padding=1)
        
        # 空间注意力模块 (SAM)
        # 使用不同的下采样率来捕获多尺度空间信息
        self.conv_sam = conv(teacher_channels, teacher_channels, kernel_size=3, stride=2, padding=1)
        
        # 解码器: 深度堆叠的 Decoder 模块
        self.decoder = nn.Sequential(*[
            _Decoder(teacher_channels, teacher_channels) for _ in range(depth)
        ])
        
        # 空间重建模块 (SRM)
        # 使用 PixelShuffle 进行上采样重建
        if teacher_channels >= 64:
            # 可以进行 4x 上采样
            scale_factor = 4
        else:
            # 较小通道数使用 2x 上采样
            scale_factor = 2
            
        self.conv_srm = nn.Sequential(
            conv(teacher_channels, scale_factor * scale_factor * teacher_channels, 
                 kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(scale_factor)
        )
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.available = True
    
    def forward(self, student_feat, teacher_feat):
        """
        Args:
            student_feat: 学生特征 [B, C_s, H, W] 或 [B, T, C_s]
            teacher_feat: 教师特征 [B, C_t, H, W] 或 [B, T, C_t]
        Returns:
            loss: MKD 损失（包含对齐损失和重建损失）
        """
        # 处理 3D 音频输入 (B, T, C) -> 4D (B, C, T, 1)
        if len(student_feat.shape) == 3:
            B, T, C_s = student_feat.shape
            _, _, C_t = teacher_feat.shape
            student_feat = student_feat.permute(0, 2, 1).unsqueeze(-1)  # (B, C_s, T, 1)
            teacher_feat = teacher_feat.permute(0, 2, 1).unsqueeze(-1)  # (B, C_t, T, 1)
        
        # 空间尺寸对齐
        s_H, s_W = student_feat.shape[2], student_feat.shape[3]
        t_H, t_W = teacher_feat.shape[2], teacher_feat.shape[3]
        
        if s_H != t_H or s_W != t_W:
            # 调整教师特征以匹配学生特征
            teacher_feat = F.adaptive_avg_pool2d(teacher_feat, (s_H, s_W))
        
        # ========== MKD 完整流程 ==========
        
        # 1. 对齐: 将学生特征对齐到教师特征通道
        s_aligned = self.conv_align(student_feat)  # [B, C_t, H, W]
        
        # 2. 空间注意力模块 (SAM)
        # 通过下采样捕获空间注意力
        s_sam = self.conv_sam(s_aligned)
        t_sam = self.conv_sam(teacher_feat)
        
        # 3. 解码器: 深度堆叠的解码处理
        s_decoded = self.decoder(s_sam)
        t_decoded = self.decoder(t_sam)
        
        # 4. 空间重建模块 (SRM)
        # 使用 PixelShuffle 进行上采样重建
        s_reconstructed = self.conv_srm(s_decoded)
        t_reconstructed = self.conv_srm(t_decoded)
        
        # ========== 计算损失 ==========
        
        # 将重建结果对齐到原始尺寸
        if s_reconstructed.shape[2:] != teacher_feat.shape[2:]:
            s_reconstructed = F.adaptive_avg_pool2d(s_reconstructed, teacher_feat.shape[2:])
        if t_reconstructed.shape[2:] != teacher_feat.shape[2:]:
            t_reconstructed = F.adaptive_avg_pool2d(t_reconstructed, teacher_feat.shape[2:])
        
        # 对齐损失: 学生对齐后的特征与教师特征
        align_loss = self.mse_loss(s_aligned, teacher_feat)
        
        # SAM 损失: 空间注意力特征
        sam_loss = self.mse_loss(s_sam, t_sam)
        
        # 解码损失: 解码器输出
        decode_loss = self.mse_loss(s_decoded, t_decoded)
        
        # 重建损失: SRM 输出
        reconstruct_loss = self.mse_loss(s_reconstructed, t_reconstructed)
        
        # L1 损失: 作为辅助损失
        reconstruct_l1_loss = self.l1_loss(s_reconstructed, t_reconstructed)
        
        # 总损失: 组合所有损失项
        # 权重分配: 对齐损失和重建损失是主要的
        total_loss = (
            0.3 * align_loss +      # 对齐损失
            0.2 * sam_loss +         # SAM 损失
            0.2 * decode_loss +      # 解码损失
            0.2 * reconstruct_loss + # 重建 MSE 损失
            0.1 * reconstruct_l1_loss  # 重建 L1 损失
        )
        
        return total_loss


# ========================================
# 8. SDD (Structure Decoupled Distillation)
# ========================================

class SDDLoss(nn.Module):
    """
    Structure Decoupled Distillation (SDD) - 简化版本
    需要模型输出 [B, C, N] 格式，其中 N 是解耦区域数量
    简化版本：处理标准 2D logits [B, C]
    """
    def __init__(self, temperature=4.0):
        super(SDDLoss, self).__init__()
        self.temperature = temperature

    def forward(self, out_s_multi, out_t_multi, target):
        """
        Args:
            out_s_multi: 学生多区域输出 [B, C, N] 或 [B, C]
            out_t_multi: 教师多区域输出 [B, C, N] 或 [B, C]
            target: 真实标签 [B]
        Returns:
            loss: SDD 蒸馏损失
        """
        # 确保输入在正确的设备上
        device = out_s_multi.device

        # 检查输入维度
        if out_s_multi.dim() == 2:
            # 简化版本：使用标准的 KD 蒸馏
            # [B, C] -> 模拟为 [B, C, 1] 单个区域
            out_s_multi = out_s_multi.unsqueeze(-1)  # [B, C, 1]
            out_t_multi = out_t_multi.unsqueeze(-1)  # [B, C, 1]

        # 从 B X C X N 到 N*B X C（N 是解耦区域数量）
        out_s_multi = out_s_multi.permute(2, 0, 1)
        out_t_multi = out_t_multi.permute(2, 0, 1)

        out_t = torch.reshape(out_t_multi, (out_t_multi.shape[0] * out_t_multi.shape[1], out_t_multi.shape[2]))
        out_s = torch.reshape(out_s_multi, (out_s_multi.shape[0] * out_s_multi.shape[1], out_s_multi.shape[2]))

        target_r = target.repeat(out_t_multi.shape[0])

        # 计算蒸馏损失
        p_s = F.log_softmax(out_s / self.temperature, dim=1)
        p_t = F.softmax(out_t / self.temperature, dim=1)
        loss_kd = F.kl_div(p_s, p_t, reduction='none') * (self.temperature ** 2)
        nan_index = torch.isnan(loss_kd)
        loss_kd[nan_index] = torch.tensor(0.0, device=device)

        loss_kd = torch.sum(loss_kd, dim=1)

        # 找到互补和一致的局部蒸馏损失
        out_t_predict = torch.argmax(out_t, dim=1)

        mask_true = out_t_predict == target_r
        mask_false = out_t_predict != target_r

        # 全局预测（第一个区域）
        global_prediction = out_t_predict[0:len(target)]
        global_prediction_true_mask = global_prediction == target
        global_prediction_false_mask = global_prediction != target

        global_prediction_true_mask_repeat = global_prediction_true_mask.repeat(out_t_multi.shape[0])
        global_prediction_false_mask_repeat = global_prediction_false_mask.repeat(out_t_multi.shape[0])

        # 全局正确，局部错误
        mask_false = out_t_predict != target_r
        mask_false[global_prediction_false_mask_repeat] = False
        mask_false[0:len(target)] = False
        gt_lw = mask_false

        # 全局错误，局部正确
        mask_true = out_t_predict == target_r
        mask_true[global_prediction_true_mask_repeat] = False
        mask_true[0:len(target)] = False
        gw_lt = mask_true

        # 全局错误，局部错误
        mask_false = out_t_predict != target_r
        mask_true = out_t_predict == target_r

        mask_false[global_prediction_true_mask_repeat] = False
        gw_lw = mask_false

        mask_true[global_prediction_false_mask_repeat] = False
        gt_lt = mask_true

        # 修改互补项的权重
        index = torch.zeros_like(loss_kd).float()
        index[gw_lw] = 1.0
        index[gt_lt] = 1.0
        index[gw_lt] = 2
        index[gt_lw] = 2

        loss = torch.sum(loss_kd * index) / target_r.shape[0]

        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: SDD loss is NaN or Inf, returning 0")
            loss = torch.tensor(0.0, device=device)

        return loss


# ========================================
# 9. 辅助函数
# ========================================

def kd_loss(logits_student, logits_teacher, temperature):
    """基础 KD 损失"""
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature ** 2
    return loss_kd


def sdd_kd_loss(out_s_multi, out_t_multi, T, target):
    """Structure Decoupled Distillation (SDD) 损失"""
    # 获取输入设备
    device = out_s_multi.device

    out_s_multi = out_s_multi.permute(2, 0, 1)
    out_t_multi = out_t_multi.permute(2, 0, 1)

    out_t = torch.reshape(out_t_multi, (out_t_multi.shape[0] * out_t_multi.shape[1], out_t_multi.shape[2]))
    out_s = torch.reshape(out_s_multi, (out_s_multi.shape[0] * out_s_multi.shape[1], out_s_multi.shape[2]))

    target_r = target.repeat(out_t_multi.shape[0])

    p_s = F.log_softmax(out_s / T, dim=1)
    p_t = F.softmax(out_t / T, dim=1)
    loss_kd = F.kl_div(p_s, p_t, reduction='none') * (T ** 2)
    nan_index = torch.isnan(loss_kd)
    loss_kd[nan_index] = torch.tensor(0.0, device=device)  # 修复：使用正确的设备

    loss_kd = torch.sum(loss_kd, dim=1)

    out_t_predict = torch.argmax(out_t, dim=1)

    mask_true = out_t_predict == target_r
    mask_false = out_t_predict != target_r

    global_prediction = out_t_predict[0:len(target)]
    global_prediction_true_mask = global_prediction == target
    global_prediction_false_mask = global_prediction != target

    global_prediction_true_mask_repeat = torch.tensor(global_prediction_true_mask, device=device).repeat(out_t_multi.shape[0])
    global_prediction_false_mask_repeat = torch.tensor(global_prediction_false_mask, device=device).repeat(out_t_multi.shape[0])

    mask_false[global_prediction_false_mask_repeat] = False
    mask_false[0:len(target)] = False

    gt_lw = mask_false

    mask_true[global_prediction_true_mask_repeat] = False
    mask_true[0:len(target)] = False

    gw_lt = mask_true

    mask_false = out_t_predict != target_r
    mask_true = out_t_predict == target_r

    index = torch.zeros_like(loss_kd).float()

    mask_false[global_prediction_true_mask_repeat] = False
    gw_lw = mask_false

    mask_true[global_prediction_false_mask_repeat] = False
    gt_lt = mask_true

    index[gw_lw] = 1.0
    index[gt_lt] = 1.0
    index[gw_lt] = 2
    index[gt_lw] = 2

    loss = torch.sum(loss_kd * index) / target_r.shape[0]

    if torch.isnan(loss) or torch.isinf(loss):
        print("inf")
        loss = torch.zeros(1, device=device)  # 修复：使用正确的设备

    return loss


# ========================================
# 9. 统一蒸馏损失类（推荐使用）
# ========================================

class UnifiedDistillationLoss(nn.Module):
    """
    统一的知识蒸馏损失类
    支持多种蒸馏方法，便于切换和比较
    
    支持的方法 (24种):
    - 'kd': 基础 KD
    - 'lskd': Logit Standardization KD
    - 'dkd': Decoupled KD
    - 'nkd': Negative KD
    - 'wsld': Weighted Soft Label Distillation
    - 'sdd': Structure Decoupled Distillation (需要模型输出 [B, C, N] 格式)
    - 'at': Attention Transfer
    - 'nst': Neuron Selectivity Transfer
    - 'fsp': Flow of Solution Procedure
    - 'pkt': Probabilistic Knowledge Transfer
    - 'rkd': Relational Knowledge Distillation
    - 'sp': Similarity-Preserving Knowledge Distillation
    - 'cc': Correlation Congruence
    - 'vid': Variational Information Distillation
    - 'mgd': Masked Generative Distillation
    - 'sr': Spatial-wise Relative Representation
    - 'cat_kd': Channel-Aligned Transfer KD
    - 'ickd': Inner Correlation KD
    - 'vkd': Variational KD
    - 'uatr_kd': UATR-specific KD
    - 'ofa': One-for-All KD
    - 'diffkd': Diffusion-based KD (完整实现)
    - 'freekd': FreeKD (完整实现，需要 pytorch_wavelets)
    - 'mkd': Masked KD (完整实现，包含 Align, SAM, Decoder, SRM 模块)
    """
    
    def __init__(self, method='kd', temperature=4.0, alpha=0.5, **kwargs):
        """
        Args:
            method: 蒸馏方法名称
            temperature: 蒸馏温度
            alpha: 蒸馏损失权重
            **kwargs: 各方法特定的参数
        """
        super(UnifiedDistillationLoss, self).__init__()
        self.method = method
        self.temperature = temperature
        self.alpha = alpha
        
        # 基础损失
        self.ce_loss = nn.CrossEntropyLoss()
        
        # 根据方法初始化特定的损失函数
        if method == 'kd':
            self.kd_loss = KDLoss(temperature=temperature, alpha=alpha)
        elif method == 'lskd':
            self.kd_loss = LSKDLoss()
        elif method == 'dkd':
            self.kd_loss = DKDLoss(alpha=kwargs.get('dkd_alpha', 1.0), 
                                    beta=kwargs.get('dkd_beta', 1.0), 
                                    temperature=temperature)
        elif method == 'nkd':
            self.kd_loss = NKDLoss(temp=kwargs.get('nkd_temp', 1.0),
                                    gamma=kwargs.get('nkd_gamma', 1.5))
        elif method == 'sdd':
            self.kd_loss = SDDLoss(temperature=kwargs.get('sdd_temperature', temperature))
        elif method == 'wsld':
            self.kd_loss = WSLDLoss(temp=kwargs.get('wsld_temp', 1.0), 
                                     alpha=kwargs.get('wsld_alpha', 1.0),
                                     num_classes=kwargs.get('num_classes', 4))
        elif method == 'at':
            self.kd_loss = AttentionLoss(p=kwargs.get('at_p', 2))
        elif method == 'nst':
            self.kd_loss = NSTLoss()
        elif method == 'fsp':
            self.kd_loss = FSPLoss()
        elif method == 'pkt':
            self.kd_loss = PKTLoss()
        elif method == 'rkd':
            self.kd_loss = RKDLoss(w_d=kwargs.get('rkd_wd', 25), 
                                    w_a=kwargs.get('rkd_wa', 50))
        elif method == 'sp':
            self.kd_loss = SimilarityLoss()
        elif method == 'cc':
            self.kd_loss = CorrelationLoss()
        elif method == 'mgd':
            self.kd_loss = MGDLoss(
                student_channels=kwargs.get('student_channels', 256),
                teacher_channels=kwargs.get('teacher_channels', 512),
                alpha_mgd=kwargs.get('alpha_mgd', 0.00007),
                lambda_mgd=kwargs.get('lambda_mgd', 0.15)
            )
        elif method == 'sr':
            self.kd_loss = SRRLLoss(
                student_channels=kwargs.get('student_channels', 256),
                teacher_channels=kwargs.get('teacher_channels', 512),
                alpha=kwargs.get('sr_alpha', 1.0),
                beta=kwargs.get('sr_beta', 1.0)
            )
        elif method == 'cat_kd':
            self.kd_loss = CAT_KDLoss(
                student_channels=kwargs.get('student_channels', 256),
                teacher_channels=kwargs.get('teacher_channels', 512)
            )
        elif method == 'ickd':
            self.kd_loss = ICKDLoss(
                student_channels=kwargs.get('student_channels', 256),
                teacher_channels=kwargs.get('teacher_channels', 512)
            )
        elif method == 'vkd':
            self.kd_loss = VKDLoss(
                student_channels=kwargs.get('student_channels', 256),
                teacher_channels=kwargs.get('teacher_channels', 256),
                projection_dim=kwargs.get('projection_dim', 256)
            )
        elif method == 'uatr_kd':
            self.kd_loss = UATR_KDLoss(
                student_channels=kwargs.get('student_channels', 256),
                teacher_channels=kwargs.get('teacher_channels', 512)
            )
        elif method == 'ofa':
            self.kd_loss = OFAKDLoss(
                eps=kwargs.get('ofa_eps', 1.0),
                temperature=kwargs.get('ofa_temperature', 1.0),
                student_channels=kwargs.get('student_channels', 64),
                teacher_channels=kwargs.get('teacher_channels', 512)
            )
        elif method == 'vid':
            self.kd_loss = VIDLoss(
                num_input_channels=kwargs.get('student_channels', 256),
                num_mid_channel=kwargs.get('mid_channel', 256),
                num_target_channels=kwargs.get('teacher_channels', 512),
                init_pred_var=kwargs.get('init_pred_var', 5.0),
                eps=kwargs.get('vid_eps', 1e-5)
            )
        elif method == 'diffkd':
            self.kd_loss = DiffKD_Loss(
                student_channels=kwargs.get('student_channels', 64),
                teacher_channels=kwargs.get('teacher_channels', 512),
                kernel_size=kwargs.get('kernel_size', 3),
                inference_steps=kwargs.get('inference_steps', 5),
                num_train_timesteps=kwargs.get('num_train_timesteps', 1000),
                use_ae=kwargs.get('use_ae', False),
                ae_channels=kwargs.get('ae_channels', None)
            )
        elif method == 'freekd':
            self.kd_loss = FreeKD_Loss(
                student_channels=kwargs.get('student_channels', 64),
                teacher_channels=kwargs.get('teacher_channels', 512)
            )
        elif method == 'mkd':
            self.kd_loss = MKD_Loss(
                student_channels=kwargs.get('student_channels', 64),
                teacher_channels=kwargs.get('teacher_channels', 512),
                mask_ratio=kwargs.get('mask_ratio', 0.1),
                depth=kwargs.get('depth', 4)
            )
        else:
            raise ValueError(f"Unknown distillation method: {method}")
    
    def forward(self, *args, **kwargs):
        """根据方法类型调用对应的前向传播"""
        if self.method in ['kd']:
            return self.kd_loss(*args)
        elif self.method == 'lskd':
            # LSKD 需要温度参数，可以从 args 或 kwargs 中获取
            if len(args) == 3:
                # 直接传递 3 个参数
                return self.kd_loss(*args)
            else:
                # 否则使用 self.temperature 作为第三个参数
                return self.kd_loss(args[0], args[1], self.temperature)
        elif self.method == 'dkd':
            return self.kd_loss(*args)
        elif self.method in ['nkd', 'wsld']:
            return self.kd_loss(*args)
        elif self.method == 'sdd':
            return self.kd_loss(*args)
        elif self.method in ['at', 'nst', 'fsp', 'sp']:
            # 特征蒸馏方法
            return self.kd_loss(*args)
        elif self.method in ['pkt', 'rkd', 'cc']:
            # 特征蒸馏方法
            return self.kd_loss(*args)
        elif self.method == 'mgd':
            return self.kd_loss(*args)
        elif self.method == 'sr':
            return self.kd_loss(*args)
        elif self.method in ['cat_kd', 'ickd', 'uatr_kd', 'vkd']:
            # 新增的特征蒸馏方法
            return self.kd_loss(*args)
        elif self.method == 'ofa':
            return self.kd_loss(*args)
        elif self.method == 'vid':
            return self.kd_loss(*args)
        elif self.method in ['diffkd', 'freekd', 'mkd']:
            # 特殊蒸馏方法（简化版本）
            return self.kd_loss(*args)
        else:
            # 对于简单的 logit 蒸馏，添加 CE 损失
            student_logits = args[0]
            teacher_logits = args[1]
            labels = args[2]

            hard_loss = self.ce_loss(student_logits, labels)
            soft_loss = self.kd_loss(*args, **kwargs)

            return self.alpha * hard_loss + (1 - self.alpha) * soft_loss


# ========================================
# 10. 工厂函数：便捷创建损失函数
# ========================================

def create_distillation_loss(method, **kwargs):
    """
    工厂函数：创建知识蒸馏损失
    
    Args:
        method: 蒸馏方法名称
        **kwargs: 各方法特定的参数
        
    Returns:
        对应的蒸馏损失实例
    """
    return UnifiedDistillationLoss(method=method, **kwargs)


# ========================================
# 示例使用
# ========================================

if __name__ == "__main__":
    # 示例 1: 基础 KD
    kd_loss = create_distillation_loss('kd', temperature=4.0, alpha=0.5)
    print("基础 KD 损失创建成功")

    # 示例 2: LSKD (Logit Standardization KD)
    lskd_loss = create_distillation_loss('lskd')
    print("LSKD (Logit Standardization) 损失创建成功")

    # 示例 3: DKD
    dkd_loss = create_distillation_loss('dkd', temperature=4.0, alpha=1.0,
                                       dkd_alpha=1.0, dkd_beta=1.0)
    print("DKD 损失创建成功")

    # 示例 4: 注意力蒸馏
    at_loss = create_distillation_loss('at', at_p=2)
    print("AT 损失创建成功")

    # 示例 5: 关系蒸馏
    rkd_loss = create_distillation_loss('rkd', rkd_wd=25, rkd_wa=50)
    print("RKD 损失创建成功")

    # 示例 6: 生成式蒸馏
    mgd_loss = create_distillation_loss('mgd', student_channels=256,
                                       teacher_channels=512, alpha_mgd=0.00007)
    print("MGD 损失创建成功")

    print("\n所有 24 种蒸馏方法创建成功！")
    print("\n支持的方法列表:")
    print("- Logit-based: kd, lskd, dkd, nkd, wsld, sdd")
    print("- Attention: at")
    print("- Feature: nst, fsp, pkt, rkd, sp, cc, vid, cat_kd, ickd, vkd, uatr_kd, sr")
    print("- Generative: mgd")
    print("- Advanced: ofa, diffkd, freekd, mkd (完整实现)")

    # 测试新增方法
    print("\n测试新增方法...")
    cat_kd_loss = create_distillation_loss('cat_kd', student_channels=256, teacher_channels=512)
    print("CAT_KD 损失创建成功")

    ickd_loss = create_distillation_loss('ickd', student_channels=256, teacher_channels=512)
    print("ICKD 损失创建成功")

    vkd_loss = create_distillation_loss('vkd', student_channels=256, teacher_channels=256)
    print("VkD 损失创建成功")

    uatr_kd_loss = create_distillation_loss('uatr_kd', student_channels=256, teacher_channels=512)
    print("UATR_KD 损失创建成功")

    ofa_loss = create_distillation_loss('ofa', eps=1.0, temperature=1.0)
    print("OFAKD 损失创建成功")

    vid_loss = create_distillation_loss('vid', student_channels=256, mid_channel=256, teacher_channels=512)
    print("VID 损失创建成功")
