#!/usr/bin/env python3
"""
批量运行蒸馏方法的训练脚本
基于 run_24_distillation_methods_shipsear.py 的修改版本
适用于 ShipsEar 数据集

修改:
- 只保留指定的21种蒸馏方法 (6 Logit-based + 15 Feature-based)
- 添加 AUC 和 F1_SCORE 评价指标

包含的蒸馏方法:
Logit-based (6种):
  1. KD - 基础知识蒸馏
  2. LSKD - Logit标准化蒸馏
  3. DKD - 解耦知识蒸馏
  4. NKD - 负知识蒸馏
  5. WSLD - 加权软标签蒸馏
  6. FreeKD - 自由蒸馏 (使用小波变换)

Feature-based (15种):
  1. AT - 注意力迁移
  2. NST - 神经元选择性迁移
  3. FSP - 求解流程
  4. PKT - 概率知识迁移
  5. RKD - 关系知识蒸馏
  6. SP - 相似性保持
  7. CC - 相关性一致性
  8. VID - 变分信息蒸馏
  9. CAT_KD - 通道对齐蒸馏
  10. ICKD - 内部相关性蒸馏
  11. VkD - 变分蒸馏
  12. MGD - 掩码生成蒸馏
  13. SDD - 结构解耦蒸馏
  14. DiffKD - 扩散蒸馏
  15. MKD - 掩码蒸馏 (完整实现)

评价指标:
  - Accuracy (准确率)
  - AUC (曲线下面积)
  - F1_SCORE (F1分数)
"""

import os
import argparse
import yaml
import logging
from datetime import datetime
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# 添加 AUC 和 F1_SCORE 支持
from sklearn.metrics import roc_auc_score, f1_score

from models.Audio_TeacherNet import build_Audio_TeacherNet
from models.LNN import AudioCfC
from models.distillation import AudioDistillationModel
from datasets.audio_dataset import AudioDataset, train_transform, validation_test_transform

# 导入统一的蒸馏损失函数
import sys
sys.path.append('experiments/1_comparison')
from distillation_loss import UnifiedDistillationLoss, create_distillation_loss


# ========================================
# 指定的 20 种蒸馏方法
# ========================================
DISTILL_METHODS = [
    # Logit-based 方法 (6种)
    "kd",          # 1. 基础知识蒸馏
    "lskd",        # 2. Logit标准化蒸馏
    "dkd",         # 3. 解耦知识蒸馏
    "nkd",         # 4. 负知识蒸馏
    "wsld",        # 5. 加权软标签蒸馏
    "freekd",      # 6. 自由蒸馏 (使用小波变换)

    # Feature-based 方法 (15种)
    "at",          # 1. 注意力迁移
    "nst",         # 2. 神经元选择性迁移
    "fsp",         # 3. 求解流程
    "pkt",         # 4. 概率知识迁移
    "rkd",         # 5. 关系知识蒸馏
    "sp",          # 6. 相似性保持
    "cc",          # 7. 相关性一致性
    "vid",         # 8. 变分信息蒸馏
    "cat_kd",      # 9. 通道对齐蒸馏
    "ickd",        # 10. 内部相关性蒸馏
    "vkd",         # 11. 变分蒸馏
    "mgd",         # 12. 掩码生成蒸馏
    "sdd",         # 13. 结构解耦蒸馏
    "diffkd",      # 14. 扩散蒸馏
    "mkd",         # 15. 掩码蒸馏 (完整实现)
]

# 总共 21 种方法 (6 Logit-based + 15 Feature-based)
# 注意: 脚本文件名为 run_20_distillation_methods.py,但实际上运行的是21种方法


# ========================================
# 工具函数（从原文件复制）
# ========================================
def setup_logger(save_dir: str, rank: int) -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger('comparison_distillation')
    logger.setLevel(logging.DEBUG)  # 设置为 DEBUG 级别以看到详细信息
    logger.handlers = []

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(save_dir, f'train_{timestamp}.log')

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 文件也记录 DEBUG

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)  # 控制台也显示 DEBUG

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    else:
        logger.addHandler(logging.NullHandler())

    return logger


def find_free_port():
    """查找一个可用的端口"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def setup_ddp(rank: int, world_size: int, config: dict, gpu_ids: list, port: int = None):
    """初始化 DDP"""
    os.environ['MASTER_ADDR'] = config['distributed']['master_addr']
    
    # 如果没有指定端口,则使用配置文件中的端口
    if port is None:
        port = config['distributed']['master_port']
    
    # 端口保持为字符串或整数
    os.environ['MASTER_PORT'] = str(port)
    print(f"Rank {rank}: Using port {port}")
    
    dist.init_process_group(
        backend=config['distributed']['backend'],
        rank=rank,
        world_size=world_size
    )
    local_gpu_id = gpu_ids[rank]
    torch.cuda.set_device(local_gpu_id)


def cleanup_ddp():
    """清理 DDP"""
    dist.destroy_process_group()


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 确保数值类型正确
    config['training']['lr'] = float(config['training']['lr'])
    config['training']['weight_decay'] = float(config['training']['weight_decay'])
    config['distillation']['temperature'] = float(config['distillation']['temperature'])
    config['distillation']['alpha'] = float(config['distillation']['alpha'])

    return config


def save_checkpoint(model: nn.Module, optimizer, scheduler, epoch: int,
                    best_acc: float, best_auc: float, best_f1: float,
                    save_path: str, is_best: bool = False):
    """保存检查点"""
    if isinstance(model, DDP):
        student_state = model.module.student.state_dict()
    else:
        student_state = model.student.state_dict()

    checkpoint = {
        'epoch': epoch,
        'student_state_dict': student_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
        'best_auc': best_auc,
        'best_f1': best_f1
    }

    torch.save(checkpoint, save_path)

    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), 'best_student.pth')
        torch.save({
            'student_state_dict': student_state,
            'best_acc': best_acc,
            'best_auc': best_auc,
            'best_f1': best_f1
        }, best_path)


def auto_resume_if_possible(model, optimizer, scheduler, save_dir, logger):
    """自动检测并恢复最近checkpoint"""
    import glob

    ckpts = glob.glob(os.path.join(save_dir, "checkpoint_epoch_*.pth"))
    if len(ckpts) == 0:
        logger.info("No checkpoint found, start from scratch")
        return 1, 0.0, 0.0, 0.0  # 返回 start_epoch, best_acc, best_auc, best_f1

    ckpts.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    latest_ckpt = ckpts[-1]

    logger.info(f"🔥 Resume from: {latest_ckpt}")

    checkpoint = torch.load(latest_ckpt, map_location="cpu")

    # 加载模型权重
    if isinstance(model, DDP):
        model.module.student.load_state_dict(checkpoint['student_state_dict'])
    else:
        model.student.load_state_dict(checkpoint['student_state_dict'])

    # 尝试加载优化器状态（可能失败，例如参数组不匹配）
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Optimizer state loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load optimizer state: {e}")
        logger.warning("Continuing with fresh optimizer state (model weights loaded)")

    # 尝试加载调度器状态（也可能失败）
    try:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info("Scheduler state loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load scheduler state: {e}")
        logger.warning("Continuing with fresh scheduler state")

    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint.get('best_acc', 0.0)
    best_auc = checkpoint.get('best_auc', 0.0)
    best_f1 = checkpoint.get('best_f1', 0.0)

    logger.info(f"Resume epoch: {start_epoch}")
    logger.info(f"Best acc: {best_acc}")

    return start_epoch, best_acc, best_auc, best_f1


def get_inputs(input_data, data_type, device):
    """根据数据类型获取输入"""
    if '@' not in data_type:
        return input_data[0].to(device)
    else:
        return [data.to(device) for data in input_data]


# ========================================
# 训练器类（从原文件复制）
# ========================================
class ComparisonDistillationTrainer:
    """对比蒸馏训练器"""
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer,
        scheduler,
        device: torch.device,
        config: dict,
        logger: logging.Logger,
        rank: int = 0,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.logger = logger
        self.rank = rank
        self.best_acc = 0.0
        self.best_auc = 0.0
        self.best_f1 = 0.0
        self.data_type = config['dataset']['data_type']
        self.distill_method = config['distillation']['method']

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> dict:
        """训练一个 epoch"""
        self.model.train()

        # 教师网络保持 eval 模式
        is_teacher_training = self.config['model']['teacher'].get('freeze', True) == False
        if not is_teacher_training:
            if isinstance(self.model, DDP):
                self.model.module.teacher.eval()
            else:
                self.model.teacher.eval()

        total_loss = 0.0
        total_hard_loss = 0.0
        total_soft_loss = 0.0
        correct = 0
        total = 0
        valid_batches = 0

        for batch_idx, (input_data, labels) in enumerate(train_loader):
            inputs = get_inputs(input_data, self.data_type, self.device)
            labels = labels.to(self.device)

            # 前向传播
            student_logits, stu_seq_logits, _, _, _, _, x_encoder, \
                teacher_logits, output_cnn_features, teacher_all_hidden_states = self.model(inputs)

            # 检查输出是否有异常值
            if torch.isnan(student_logits).any() or torch.isinf(student_logits).any():
                self.logger.warning(f'NaN/Inf in outputs at epoch {epoch}, batch {batch_idx}')
                continue

            # 根据蒸馏方法计算损失
            try:
                if self.distill_method in ['kd', 'lskd', 'dkd', 'nkd', 'wsld']:
                    # Logit-based 蒸馏方法 (6种: kd, lskd, dkd, nkd, wsld, freekd)
                    loss, hard_loss, soft_loss = self._compute_logit_distillation_loss(
                        student_logits, stu_seq_logits, teacher_logits, labels
                    )
                elif self.distill_method in ['at', 'nst', 'fsp', 'pkt', 'rkd', 'sp', 'cc', 'vid', 'freekd']:
                    # 特征蒸馏方法 (15种: at, nst, fsp, pkt, rkd, sp, cc, vid, freekd, cat_kd, ickd, vkd, mgd, sdd, diffkd, mkd)
                    loss, hard_loss, soft_loss = self._compute_feature_distillation_loss(
                        student_logits, stu_seq_logits, x_encoder,
                        teacher_logits, output_cnn_features, teacher_all_hidden_states,
                        labels
                    )
                elif self.distill_method in ['cat_kd', 'ickd', 'uatr_kd', 'vkd', 'mgd', 'sr']:
                    # 通道/对齐蒸馏方法
                    loss, hard_loss, soft_loss = self._compute_channel_distillation_loss(
                        student_logits, stu_seq_logits, x_encoder,
                        teacher_logits, output_cnn_features,
                        labels
                    )
                elif self.distill_method in ['sdd', 'diffkd', 'mkd']:
                    # 特殊蒸馏方法 (需要特殊处理)
                    loss, hard_loss, soft_loss = self._compute_special_distillation_loss(
                        student_logits, stu_seq_logits, x_encoder,
                        teacher_logits, output_cnn_features, teacher_all_hidden_states,
                        labels
                    )
                else:
                    raise ValueError(f"Unknown distillation method: {self.distill_method}")
            except Exception as e:
                self.logger.error(f'Error computing loss for {self.distill_method} at epoch {epoch}, batch {batch_idx}: {e}')
                import traceback
                self.logger.error(traceback.format_exc())
                continue

            # 检查 loss 是否有异常值
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(f'Invalid loss at epoch {epoch}, batch {batch_idx}')
                continue

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            total_hard_loss += hard_loss  # hard_loss 已经是 float
            total_soft_loss += soft_loss  # soft_loss 已经是 float

            pred = student_logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            valid_batches += 1

            if self.rank == 0 and batch_idx % self.config['training']['log_interval'] == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(
                    f'Epoch [{epoch}] Batch [{batch_idx}/{len(train_loader)}] '
                    f'Loss: {loss.item():.4f}, Hard: {hard_loss:.4f}, Soft: {soft_loss:.4f} '
                    f'Acc: {100.*correct/total:.2f}% LR: {current_lr:.6f}'
                )

        metrics = {
            'loss': total_loss / max(valid_batches, 1),
            'hard_loss': total_hard_loss / max(valid_batches, 1),
            'soft_loss': total_soft_loss / max(valid_batches, 1),
            'accuracy': correct / max(total, 1)
        }

        return metrics

    def _compute_logit_distillation_loss(self, student_logits, stu_seq_logits,
                                         teacher_logits, labels):
        """计算 Logit-based 蒸馏损失"""
        # 硬标签损失
        hard_loss = F.cross_entropy(student_logits, labels)

        # 蒸馏损失
        if self.distill_method == 'kd':
            # KD: criterion 返回的是 total_loss (alpha * hard_loss + (1-alpha) * soft_loss)
            # 所以我们需要重新计算 soft_loss
            soft_teacher = F.softmax(teacher_logits / self.criterion.temperature, dim=1)
            soft_student = F.log_softmax(student_logits / self.criterion.temperature, dim=1)
            soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.criterion.temperature ** 2)
        elif self.distill_method == 'lskd':
            # LSKD: 使用 logit standardization
            soft_loss = self.criterion(student_logits, teacher_logits, self.criterion.temperature)
        else:
            # 其他 logit-based 方法
            soft_loss = self.criterion(student_logits, teacher_logits, labels)
            
            # 某些损失函数返回列表，需要转换为标量
            if isinstance(soft_loss, (list, tuple)):
                soft_loss = sum(soft_loss) if isinstance(soft_loss, list) else soft_loss[0]

        # 组合损失
        alpha = self.config['distillation']['alpha']
        loss = alpha * hard_loss + (1 - alpha) * soft_loss

        return loss, hard_loss.item(), soft_loss.item() if torch.is_tensor(soft_loss) else float(soft_loss)

    def _compute_feature_distillation_loss(self, student_logits, stu_seq_logits,
                                          x_encoder, teacher_logits, output_cnn_features,
                                          teacher_all_hidden_states, labels):
        """计算特征蒸馏损失"""
        # 硬标签损失
        hard_loss = F.cross_entropy(student_logits, labels)

        # 特征对齐
        # 学生特征: [B, 16, 64]
        # 教师特征: [B, 149, 512]

        # 时间维度对齐: 149 -> 16
        if output_cnn_features is not None:
            t_h = output_cnn_features.permute(0, 2, 1)  # [B, 512, 149]
            t_h = F.interpolate(t_h, size=x_encoder.size(1), mode='linear', align_corners=True)
            t_h = t_h.permute(0, 2, 1)  # [B, 16, 512]

            # PKT: 需要学生和教师特征具有相同的维度
            # 使用 teacher_linear 将教师特征从 512 维投影到 64 维（与学生特征相同）
            if self.distill_method == 'pkt':
                student_for_pkt = x_encoder  # [B, 16, 64]
                # 使用 no_grad 避免梯度传播，因为 teacher_linear 已被冻结
                with torch.no_grad():
                    teacher_for_pkt = self.model.module.teacher_linear(t_h)  # [B, 16, 512] -> [B, 16, 64]
                soft_loss = self.criterion(student_for_pkt, teacher_for_pkt)
            # FreeKD: 使用原始特征 (不投影)
            elif self.distill_method == 'freekd':
                # FreeKD 期望 [B, C, H, W] 格式
                # 将 3D [B, T, C] 转换为 4D [B, C, T, 1]
                student_for_freekd = x_encoder.permute(0, 2, 1).unsqueeze(-1)  # [B, 64, 16, 1]
                teacher_for_freekd = t_h.permute(0, 2, 1).unsqueeze(-1)  # [B, 512, 16, 1]
                soft_loss = self.criterion(student_for_freekd, teacher_for_freekd)
            else:
                # 其他特征蒸馏方法: 投影学生特征到教师特征维度
                student_projected = self.model.module.stu_linear(x_encoder)  # [B, 16, num_classes]
                soft_loss = self.criterion(student_projected, t_h)

            # 某些损失函数返回列表，需要转换为标量
            if isinstance(soft_loss, (list, tuple)):
                soft_loss = sum(soft_loss) if isinstance(soft_loss, list) else soft_loss[0]
        else:
            soft_loss = torch.tensor(0.0, device=self.device)

        # 组合损失
        alpha = self.config['distillation']['alpha']
        loss = alpha * hard_loss + (1 - alpha) * soft_loss

        return loss, hard_loss.item(), soft_loss.item() if torch.is_tensor(soft_loss) else float(soft_loss)

    def _compute_channel_distillation_loss(self, student_logits, stu_seq_logits,
                                          x_encoder, teacher_logits, output_cnn_features, labels):
        """计算通道蒸馏损失"""
        # 硬标签损失
        hard_loss = F.cross_entropy(student_logits, labels)

        # 通道对齐蒸馏
        if output_cnn_features is not None:
            # 时间对齐
            t_h = output_cnn_features.permute(0, 2, 1)  # [B,512, 149]
            t_h = F.interpolate(t_h, size=x_encoder.size(1), mode='linear', align_corners=True)
            t_h = t_h.permute(0, 2, 1)  # [B, 16, 512]

            # 根据方法选择适当的特征
            # 对于 cat_kd, ickd, vkd, mgd 等，需要确保维度匹配
            # 这些方法通常期望相同维度的特征，或使用特定的投影
            if self.distill_method in ['cat_kd', 'ickd', 'vkd', 'mgd']:
                # CAT_KD, ICKD, VkD, MGD 需要 4D 输入 [B, C, H, W]
                # 学生特征: [B, 16, 64] -> [B, 64, 16, 1]
                # 教师特征: [B, 16, 512] -> [B, 512, 16, 1]
                student_4d = x_encoder.permute(0, 2, 1).unsqueeze(-1)  # [B, 64, 16, 1]
                teacher_4d = t_h.permute(0, 2, 1).unsqueeze(-1)  # [B, 512, 16, 1]
                soft_loss = self.criterion(student_4d, teacher_4d)
            else:
                # 其他通道对齐方法
                # 对于某些方法，可能需要调整维度
                # 临时使用学生投影特征
                student_projected = self.model.module.stu_linear(x_encoder)  # [B, 16, num_classes]
                soft_loss = self.criterion(student_projected, t_h)
            
            # 某些损失函数返回列表，需要转换为标量
            if isinstance(soft_loss, (list, tuple)):
                soft_loss = sum(soft_loss) if isinstance(soft_loss, list) else soft_loss[0]
        else:
            soft_loss = torch.tensor(0.0, device=self.device)

        # 组合损失
        alpha = self.config['distillation']['alpha']
        loss = alpha * hard_loss + (1 - alpha) * soft_loss

        return loss, hard_loss.item(), soft_loss.item() if torch.is_tensor(soft_loss) else float(soft_loss)

    def _compute_special_distillation_loss(self, student_logits, stu_seq_logits,
                                         x_encoder, teacher_logits, output_cnn_features,
                                         teacher_all_hidden_states, labels):
        """计算特殊蒸馏损失 (SDD, DiffKD, MKD)"""
        # 硬标签损失
        hard_loss = F.cross_entropy(student_logits, labels)

        # 根据方法计算蒸馏损失
        if self.distill_method in ['sdd', 'diffkd', 'mkd']:
            # 这些方法使用特征蒸馏
            if output_cnn_features is not None:
                # 时间维度对齐: 149 -> 16
                t_h = output_cnn_features.permute(0, 2, 1)  # [B,512, 149]
                t_h = F.interpolate(t_h, size=x_encoder.size(1), mode='linear', align_corners=True)
                t_h = t_h.permute(0, 2, 1)  # [B, 16, 512]

                # SDD: 需要相同维度的特征 [B, C, N] 或 [B, C]
                if self.distill_method == 'sdd':
                    # SDD 期望学生和教师特征具有相同的维度
                    # 使用学生原始特征 [B, 16, 64] 和教师特征 [B, 16, 512]
                    # SDD 会自动处理维度不匹配
                    student_for_sdd = x_encoder  # [B, 16, 64]
                    teacher_for_sdd = t_h  # [B, 16, 512]
                    soft_loss = self.criterion(student_for_sdd, teacher_for_sdd, labels)
                # DiffKD: 需要 3D 或 4D 输入 [B, T, C] 或 [B, C, H, W]
                elif self.distill_method == 'diffkd':
                    # DiffKD 期望 [B, T, C] 或 [B, C, H, W] 格式
                    # 学生特征: [B, 16, 64] (3D) 或转换为 [B, 64, 16, 1] (4D)
                    # 教师特征: [B, 16, 512] (3D) 或转换为 [B, 512, 16, 1] (4D)
                    student_for_diffkd = x_encoder.permute(0, 2, 1).unsqueeze(-1)  # [B, 64, 16, 1]
                    teacher_for_diffkd = t_h.permute(0, 2, 1).unsqueeze(-1)  # [B, 512, 16, 1]
                    soft_loss = self.criterion(student_for_diffkd, teacher_for_diffkd)
                # MKD: 需要 4D 输入 [B, C, H, W]
                elif self.distill_method == 'mkd':
                    # MKD 期望 [B, C, H, W] 格式
                    # 学生特征: [B, 16, 64] -> [B, 64, 16, 1]
                    # 教师特征: [B, 16, 512] -> [B, 512, 16, 1]
                    student_for_mkd = x_encoder.permute(0, 2, 1).unsqueeze(-1)  # [B, 64, 16, 1]
                    teacher_for_mkd = t_h.permute(0, 2, 1).unsqueeze(-1)  # [B, 512, 16, 1]
                    soft_loss = self.criterion(student_for_mkd, teacher_for_mkd)
                else:
                    soft_loss = torch.tensor(0.0, device=self.device)

                # 某些损失函数返回列表，需要转换为标量
                if isinstance(soft_loss, (list, tuple)):
                    soft_loss = sum(soft_loss) if isinstance(soft_loss, list) else soft_loss[0]
            else:
                soft_loss = torch.tensor(0.0, device=self.device)
        else:
            soft_loss = torch.tensor(0.0, device=self.device)

        # 组合损失
        alpha = self.config['distillation']['alpha']
        loss = alpha * hard_loss + (1 - alpha) * soft_loss

        return loss, hard_loss.item(), soft_loss.item() if torch.is_tensor(soft_loss) else float(soft_loss)

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict:
        """验证 - 添加 AUC 和 F1_SCORE 评价指标"""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        # 用于计算 AUC 和 F1
        all_preds = []
        all_labels = []

        for input_data, labels in val_loader:
            inputs = get_inputs(input_data, self.data_type, self.device)
            labels = labels.to(self.device)

            # 检查输入
            has_nan_inf = False
            if isinstance(inputs, list):
                for i, tensor in enumerate(inputs):
                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        has_nan_inf = True
                if has_nan_inf:
                    inputs = [torch.nan_to_num(inp, nan=0.0, posinf=1.0, neginf=-1.0) for inp in inputs]
            else:
                if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1.0, neginf=-1.0)

            # 仅使用学生网络
            student_logits, stu_seq_logits, fl, fg, bl, bg, x_encoder, \
                teacher_logits, output_cnn_features, teacher_all_hidden_states = self.model(inputs)

            # 检查输出
            if torch.isnan(student_logits).any() or torch.isinf(student_logits).any():
                continue

            loss = F.cross_entropy(student_logits, labels)

            total_loss += loss.item()
            pred = student_logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            # 保存预测和标签用于计算 AUC 和 F1
            all_preds.append(student_logits.detach().cpu())
            all_labels.append(labels.cpu())

        # DDP 下聚合结果
        if dist.is_initialized():
            world_size = dist.get_world_size()
            
            # 聚合 metrics (使用 SUM)
            metrics_tensor = torch.tensor([total_loss, correct, total], device=self.device)
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            total_loss, correct, total = metrics_tensor.tolist()

            # 合并所有预测和标签
            all_preds_tensor = torch.cat(all_preds).to(self.device)
            all_labels_tensor = torch.cat(all_labels).to(self.device)

            # 收集所有进程的预测和标签 (使用 gather)
            # 首先计算每个进程的 tensor 大小
            gathered_preds = [torch.zeros_like(all_preds_tensor) for _ in range(world_size)]
            gathered_labels = [torch.zeros_like(all_labels_tensor) for _ in range(world_size)]
            
            dist.all_gather(gathered_preds, all_preds_tensor)
            dist.all_gather(gathered_labels, all_labels_tensor)

            # 拼接所有进程的数据
            all_preds = torch.cat(gathered_preds).cpu().numpy()
            all_labels = torch.cat(gathered_labels).cpu().numpy()
        else:
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()

        # 计算 AUC 和 F1
        try:
            # AUC: 使用 softmax 概率
            pred_probs = torch.softmax(torch.tensor(all_preds), dim=1).numpy()
            
            # 检查是否有足够的类别
            unique_labels = len(set(all_labels))
            unique_preds = len(set(all_preds.argmax(axis=1)))
            
            self.logger.debug(f'AUC/F1 计算信息:')
            self.logger.debug(f'  all_labels shape: {all_labels.shape}')
            self.logger.debug(f'  all_preds shape: {all_preds.shape}')
            self.logger.debug(f'  unique_labels: {unique_labels}')
            self.logger.debug(f'  unique_preds: {unique_preds}')
            self.logger.debug(f'  pred_probs shape: {pred_probs.shape}')
            
            if unique_labels < 2:
                self.logger.warning(f'只有 {unique_labels} 个类别,无法计算 AUC')
                auc = 0.0
            else:
                # 尝试计算 AUC
                try:
                    auc = roc_auc_score(all_labels, pred_probs, multi_class='ovr', average='macro')
                    self.logger.debug(f'  AUC 计算成功: {auc}')
                except Exception as e:
                    self.logger.warning(f'AUC 计算失败: {e}')
                    # 尝试使用不同的参数
                    try:
                        auc = roc_auc_score(all_labels, pred_probs, average='macro')
                        self.logger.debug(f'  AUC (简化参数): {auc}')
                    except Exception as e2:
                        self.logger.warning(f'AUC (简化参数) 也失败: {e2}')
                        auc = 0.0

            # F1: 使用预测标签
            pred_labels = all_preds.argmax(axis=1)
            try:
                f1 = f1_score(all_labels, pred_labels, average='macro')
                self.logger.debug(f'  F1 计算成功: {f1}')
            except Exception as e:
                self.logger.warning(f'F1 计算失败: {e}')
                f1 = 0.0
                
        except Exception as e:
            self.logger.warning(f'Error computing AUC/F1 (总异常): {e}')
            import traceback
            self.logger.warning(traceback.format_exc())
            auc = 0.0
            f1 = 0.0

        num_batches = len(val_loader)
        if dist.is_initialized():
            num_batches *= dist.get_world_size()

        metrics = {
            'loss': total_loss / max(num_batches, 1),
            'accuracy': 100. * correct / max(total, 1),
            'auc': auc * 100,  # 转换为百分比
            'f1_score': f1 * 100  # 转换为百分比
        }

        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        """完整训练流程"""
        num_epochs = self.config['training']['num_epochs']
        save_dir = self.config['save']['save_dir']
        save_interval = self.config['save']['save_interval']

        os.makedirs(save_dir, exist_ok=True)

        # 自动resume
        start_epoch, self.best_acc, self.best_auc, self.best_f1 = auto_resume_if_possible(
            self.model, self.optimizer, self.scheduler, save_dir, self.logger
        )

        for epoch in range(start_epoch, num_epochs + 1):
            # DDP: 设置 epoch
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)

            # 验证
            val_metrics = self.validate(val_loader)

            # 学习率调度
            self.scheduler.step()

            # 日志记录
            if self.rank == 0:
                self.logger.info(
                    f'Epoch [{epoch}/{num_epochs}] '
                    f'Train Loss: {train_metrics["loss"]:.4f}, '
                    f'Train Acc: {train_metrics["accuracy"]:.2f}%, '
                    f'Val Loss: {val_metrics["loss"]:.4f}, '
                    f'Val Acc: {val_metrics["accuracy"]:.2f}%, '
                    f'Val AUC: {val_metrics["auc"]:.2f}%, '
                    f'Val F1: {val_metrics["f1_score"]:.2f}%'
                )

                # 保存最佳模型 (使用 Accuracy)
                is_best = val_metrics['accuracy'] > self.best_acc
                if is_best:
                    self.best_acc = val_metrics['accuracy']
                    self.best_auc = val_metrics['auc']
                    self.best_f1 = val_metrics['f1_score']
                    self.logger.info(
                        f'Best model saved! '
                        f'Acc: {self.best_acc:.2f}%, '
                        f'AUC: {self.best_auc:.2f}%, '
                        f'F1: {self.best_f1:.2f}%'
                    )

                # 定期保存
                if epoch % save_interval == 0 or is_best:
                    save_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
                    save_checkpoint(
                        self.model, self.optimizer, self.scheduler,
                        epoch, self.best_acc, self.best_auc, self.best_f1,
                        save_path, is_best
                    )
                    if epoch % save_interval == 0:
                        self.logger.info(f'Checkpoint saved at epoch {epoch}')

        if self.rank == 0:
            self.logger.info('='*50)
            self.logger.info(
                f'Training finished! '
                f'Best Acc: {self.best_acc:.2f}%, '
                f'Best AUC: {self.best_auc:.2f}%, '
                f'Best F1: {self.best_f1:.2f}%'
            )
            self.logger.info('='*50)


def main_worker(rank: int, world_size: int, config: dict, gpu_ids: list):
    """单个 GPU 的训练进程"""
    setup_ddp(rank, world_size, config, gpu_ids)

    local_gpu_id = gpu_ids[rank]
    device = torch.device(f'cuda:{local_gpu_id}')

    torch.manual_seed(config['training']['seed'] + rank)
    torch.cuda.manual_seed(config['training']['seed'] + rank)

    save_dir = config['save']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logger(save_dir, rank)
    if rank == 0:
        logger.info('='*50)
        logger.info('Comparison Distillation Training (DeepShip)')
        logger.info('='*50)
        logger.info(f'GPUs: {gpu_ids}')
        logger.info(f'World size: {world_size}')
        logger.info(f'Dataset: {config["dataset"]["data_dir"]}')
        logger.info(f'Distillation method: {config["distillation"]["method"]}')
        logger.info(f'Temperature: {config["distillation"]["temperature"]}')
        logger.info(f'Alpha: {config["distillation"]["alpha"]}')
        logger.info(f'Save dir: {save_dir}')
        logger.info('='*50)

    # 创建数据集
    train_dataset = AudioDataset(
        data_dir=config['dataset']['data_dir'],
        data_flag='train',
        data_type=config['dataset']['data_type'],
        transform=train_transform
    )

    val_dataset = AudioDataset(
        data_dir=config['dataset']['data_dir'],
        data_flag='validation',
        data_type=config['dataset']['data_type'],
        transform=validation_test_transform
    )

    if rank == 0:
        logger.info(f'Train dataset size: {len(train_dataset)}')
        logger.info(f'Validation dataset size: {len(val_dataset)}')

    # DDP Sampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        num_workers=config['dataset']['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        sampler=val_sampler,
        num_workers=config['dataset']['num_workers'],
        pin_memory=True,
        drop_last=False
    )

    # 创建模型
    num_classes = config['model']['num_classes']
    model = AudioDistillationModel(
        num_classes=num_classes,
        teacher_pretrained=config['model']['teacher']['pretrained'],
        freeze_teacher=config['model']['teacher']['freeze'],
        teacher_checkpoint=config['model']['teacher']['checkpoint_path'],
        p_encoder=config['model']['student']['p_encoder'],
        p_classifier=config['model']['student']['p_classifier']
    ).to(device)

    if rank == 0:
        student_params = sum(p.numel() for p in model.student.parameters())
        teacher_params = sum(p.numel() for p in model.teacher.parameters())
        logger.info(f'Student params: {student_params:,} ({student_params/1e6:.2f}M)')
        logger.info(f'Teacher params: {teacher_params:,} ({teacher_params/1e6:.2f}M)')

    # DDP 包装
    model = DDP(model, device_ids=[local_gpu_id], find_unused_parameters=True)

    # 创建统一的蒸馏损失函数
    method = config['distillation']['method']
    kwargs = {
        'temperature': config['distillation']['temperature'],
        'alpha': config['distillation']['alpha'],
        'num_classes': num_classes,
        'student_channels': 64,  # AudioCfC 的编码器输出维度
        'teacher_channels': 512,  # Audio_TeacherNet 的 CNN 特征维度
    }

    # 根据方法添加特定参数
    if method == 'dkd':
        kwargs.update({
            'dkd_alpha': config['distillation'].get('dkd_alpha', 1.0),
            'dkd_beta': config['distillation'].get('dkd_beta', 1.0)
        })
    elif method == 'rkd':
        kwargs.update({
            'rkd_wd': config['distillation'].get('rkd_wd', 25),
            'rkd_wa': config['distillation'].get('rkd_wa', 50)
        })
    elif method == 'wsld':
        kwargs.update({
            'wsld_temp': config['distillation'].get('wsld_temp', 1.0),
            'wsld_alpha': config['distillation'].get('wsld_alpha', 1.0)
        })

    criterion = create_distillation_loss(method=method, **kwargs).to(device)

    # 优化器
    student_params = list(model.module.student.parameters())
    
    # 对于 PKT 方法，teacher_linear 只是辅助投影层，不应该参与优化
    # PKT 的损失函数是无参数的，teacher_linear 应该保持固定
    params_to_optimize = student_params

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['num_epochs']
    )

    # 训练器
    trainer = ComparisonDistillationTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        logger=logger,
        rank=rank
    )

    if rank == 0:
        logger.info(f'Start training with method: {method}...')

    # 开始训练
    try:
        trainer.fit(train_loader, val_loader)
    except Exception as e:
        if rank == 0:
            logger.error(f'Training failed with error: {e}')
            import traceback
            logger.error(traceback.format_exc())
        raise
    finally:
        # 清理 DDP (无论成功或失败都执行)
        cleanup_ddp()


def parse_gpu_ids(gpu_str: str) -> list:
    """解析 GPU ID 字符串"""
    return [int(x.strip()) for x in gpu_str.split(',')]


# ========================================
# 批量训练主函数
# ========================================
def main():
    parser = argparse.ArgumentParser(description='批量运行 24 种蒸馏方法的训练 (DeepShip 数据集)')
    parser.add_argument('--config', type=str,
                        default='configs/train_comparison_distillation_shipsear.yaml',
                        help='Path to config file')
    parser.add_argument('--gpus', type=str, default=None,
                        help='指定GPU，如 "0,1,2,3" 或 "2,3"')
    parser.add_argument('--method', type=str, default=None,
                        help='只训练特定的蒸馏方法，如 "kd"')
    parser.add_argument('--skip_completed', action='store_true', default=True,
                        help='跳过已完成的训练（默认：True）')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 确定使用的 GPU
    if args.gpus is not None:
        gpu_ids = parse_gpu_ids(args.gpus)
    elif 'gpu_ids' in config['distributed'] and config['distributed']['gpu_ids']:
        gpu_ids = config['distributed']['gpu_ids']
    else:
        gpu_ids = list(range(torch.cuda.device_count()))

    # 验证 GPU ID
    available_gpus = torch.cuda.device_count()
    for gpu_id in gpu_ids:
        if gpu_id >= available_gpus:
            raise ValueError(f'GPU {gpu_id} 不存在，可用GPU数量: {available_gpus}')

    world_size = len(gpu_ids)

    # 确定要训练的方法
    if args.method:
        methods_to_train = [args.method]
        if args.method not in DISTILL_METHODS:
            raise ValueError(f"未知的蒸馏方法: {args.method}")
    else:
        methods_to_train = DISTILL_METHODS

    # 打印开始信息
    print('='*80)
    print('批量蒸馏方法对比实验 (Shipsear数据集)')
    print('='*80)
    print(f'配置文件: {args.config}')
    print(f'使用GPU: {gpu_ids}')
    print(f'World size: {world_size}')
    print(f'总方法数: {len(methods_to_train)}')
    print(f'跳过已完成: {args.skip_completed}')
    print('='*80)
    print()

    # 统计信息
    stats = {
        'total': len(methods_to_train),
        'completed': 0,
        'skipped': 0,
        'failed': 0
    }

    # 逐个训练每个方法
    for idx, method in enumerate(methods_to_train, 1):
        print(f'[{idx}/{stats["total"]}] 开始训练方法: {method}')
        print('-'*80)

        # 检查是否已完成
        # 使用原始配置中的保存目录作为基础
        original_base_save_dir = load_config(args.config)['save']['save_dir']
        method_save_dir = original_base_save_dir.replace('/comparison_distillation',
                                                       f'/comparison_distillation_{method}')
        checkpoint_path = os.path.join(method_save_dir, 'checkpoint_epoch_200.pth')

        if args.skip_completed and os.path.exists(checkpoint_path):
            print(f'  ⚠️  {method} 已完成训练（检查点存在），跳过')
            stats['skipped'] += 1
            print()
            continue

        # 创建新的配置副本，避免修改原始配置
        method_config = copy.deepcopy(config)

        # 修改配置中的方法
        method_config['distillation']['method'] = method
        method_config['save']['save_dir'] = method_save_dir

        # 创建保存目录
        os.makedirs(method_save_dir, exist_ok=True)

        # 查找可用端口（如果配置中端口为 0）
        if method_config['distributed']['master_port'] == '0':
            free_port = find_free_port()
            method_config['distributed']['master_port'] = str(free_port)
            print(f'  📌 使用可用端口: {free_port}')

        # 启动多进程训练
        try:
            torch.multiprocessing.spawn(
                main_worker,
                args=(world_size, method_config, gpu_ids),
                nprocs=world_size,
                join=True
            )
            print(f'  ✅ {method} 训练完成')
            stats['completed'] += 1
        except KeyboardInterrupt:
            print(f'  ⚠️  {method} 训练被中断')
            stats['failed'] += 1
            raise  # 重新抛出 KeyboardInterrupt
        except Exception as e:
            print(f'  ❌ {method} 训练失败: {e}')
            import traceback
            traceback.print_exc()
            stats['failed'] += 1

        print()

    # 打印总结
    print('='*80)
    print('批量训练完成！')
    print('='*80)
    print(f'总结:')
    print(f'  总方法数: {stats["total"]}')
    print(f'  已完成: {stats["completed"]}')
    print(f'  跳过: {stats["skipped"]}')
    print(f'  失败: {stats["failed"]}')
    print('='*80)

    # 生成结果汇总
    print()
    print('生成结果汇总...')
    print()

    results = []
    # 使用原始配置中的保存目录作为基础
    original_base_save_dir = load_config(args.config)['save']['save_dir']

    for method in methods_to_train:
        method_save_dir = original_base_save_dir.replace('/comparison_distillation',
                                                       f'/comparison_distillation_{method}')

        # 查找日志文件
        log_files = list(Path(method_save_dir).glob('train_*.log'))
        if log_files:
            log_file = log_files[-1]
            best_acc = 0.0
            best_auc = 0.0
            best_f1 = 0.0
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        # 解析最佳模型信息
                        if 'Best model saved!' in line:
                            # 格式: "Best model saved! Acc: XX.XX%, AUC: XX.XX%, F1: XX.XX%"
                            parts = line.split('Acc: ')[1].split(',')
                            acc_str = parts[0].strip().rstrip('%')
                            best_acc = float(acc_str)
                            
                            if 'AUC:' in line:
                                auc_parts = [p for p in parts if 'AUC:' in p]
                                if auc_parts:
                                    auc_str = auc_parts[0].split('AUC: ')[1].strip().rstrip('%')
                                    best_auc = float(auc_str)
                            
                            if 'F1:' in line:
                                f1_parts = [p for p in parts if 'F1:' in p]
                                if f1_parts:
                                    f1_str = f1_parts[0].split('F1: ')[1].strip().rstrip('%')
                                    best_f1 = float(f1_str)
            except Exception as e:
                print(f"  读取日志文件 {log_file} 时出错: {e}")

            has_checkpoint = os.path.exists(os.path.join(method_save_dir, 'checkpoint_epoch_200.pth'))
            status = 'completed' if has_checkpoint else 'incomplete'

            results.append({
                'method': method,
                'best_acc': best_acc,
                'best_auc': best_auc,
                'best_f1': best_f1,
                'has_checkpoint': has_checkpoint,
                'status': status
            })
        else:
            results.append({
                'method': method,
                'best_acc': 0.0,
                'best_auc': 0.0,
                'best_f1': 0.0,
                'has_checkpoint': False,
                'status': 'not_started'
            })

    # 打印结果表格
    print('蒸馏方法对比结果汇总:')
    print('='*100)
    print(f"{'方法':<15} {'最佳准确率(%)':<15} {'最佳AUC(%)':<15} {'最佳F1(%)':<15} {'状态':<15} {'检查点':<15}")
    print('='*100)

    for result in results:
        checkpoint_mark = '✓' if result['has_checkpoint'] else '✗'
        print(f"{result['method']:<15} "
              f"{result['best_acc']:<15.2f} "
              f"{result['best_auc']:<15.2f} "
              f"{result['best_f1']:<15.2f} "
              f"{result['status']:<15} "
              f"{checkpoint_mark:<15}")

    print('='*100)

    # 保存为 CSV
    csv_path = original_base_save_dir.replace('/comparison_distillation',
                                         '/comparison_distillation_results.csv')
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Best Acc (%)', 'Best AUC (%)', 'Best F1 (%)', 'Status', 'Has Checkpoint'])
        for result in results:
            writer.writerow([
                result['method'],
                f"{result['best_acc']:.2f}",
                f"{result['best_auc']:.2f}",
                f"{result['best_f1']:.2f}",
                result['status'],
                result['has_checkpoint']
            ])
    print(f'结果已保存到: {csv_path}')
    print()

    # 按准确率排序
    completed_results = [r for r in results if r['status'] == 'completed']
    if completed_results:
        completed_results.sort(key=lambda x: x['best_acc'], reverse=True)
        print('已完成方法排名（按准确率降序）:')
        print('='*40)
        for i, result in enumerate(completed_results, 1):
            print(f"{i:2d}. {result['method']:<10} - {result['best_acc']:.2f}%")
        print('='*40)

    print()
    print('所有任务已完成！')


if __name__ == '__main__':
    main()
