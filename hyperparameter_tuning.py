"""
超参数分析脚本 - 针对蒸馏损失函数中的 lambda_m 和 beta

作者: 超参数分析
日期: 2026-04-27

任务：
1. 扫描 lambda_m (控制 MemKD 和 TS-T 的权重)
2. 扫描 beta (控制 TS-T 中 seq loss 和 final loss 的权重)
"""

import os
import yaml
import argparse
import csv
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from models.Audio_TeacherNet import build_Audio_TeacherNet
from models.distillation import AudioDistillationModel
from utils.distillation_loss import DistillationLoss
from datasets.audio_dataset import AudioDataset, train_transform, validation_test_transform


# ============================================
# 日志设置（与 train_distillation_shipsear.py 保持一致）
# ============================================

def setup_logger(save_dir: str, rank: int):
    """设置日志记录器"""
    import logging
    logger = logging.getLogger('hyperparameter_tuning')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # 清空已有 handlers

    # 只在主进程记录日志
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(save_dir, f'hyperparam_tuning_{timestamp}.log')

        # 文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    else:
        logger.addHandler(logging.NullHandler())

    return logger


# ============================================
# 辅助函数
# ============================================

def setup_ddp(rank: int, world_size: int, config: dict, gpu_ids: list):
    """初始化 DDP"""
    os.environ['MASTER_ADDR'] = config['distributed']['master_addr']
    os.environ['MASTER_PORT'] = config['distributed']['master_port']
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


def get_inputs(input_data, data_type, device):
    """根据数据类型获取输入"""
    if '@' not in data_type:
        # 单一数据类型
        return input_data[0].to(device)
    else:
        # 多模态输入
        return [data.to(device) for data in input_data]


# ============================================
# 核心训练函数（任务1）
# ============================================

def train_and_evaluate(
    lambda_m: float,
    beta: float,
    config: dict,
    rank: int,
    logger,
    num_epochs: int = 20,  # 默认使用较少的epoch进行快速超参数搜索
    save_model: bool = False,  # 是否保存模型
    save_dir: str = None  # 模型保存目录
) -> float:
    """
    训练并评估模型

    输入：
        lambda_m: float - 控制 MemKD 和 TS-T 的权重
        beta: float - 控制 TS-T 中 seq loss 和 final loss 的权重
        config: dict - 配置字典
        rank: int - 进程rank
        logger: 日志记录器
        num_epochs: int - 训练轮数（默认20，可调整）

    输出：
        val_acc: float - 验证集准确率
    """
    # 获取当前进程对应的实际 GPU ID
    gpu_ids = config['distributed']['gpu_ids']
    local_gpu_id = gpu_ids[rank]
    device = torch.device(f'cuda:{local_gpu_id}')

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

    # DDP Sampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False
    )

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
    model = AudioDistillationModel(
        num_classes=config['model']['num_classes'],
        teacher_pretrained=config['model']['teacher']['pretrained'],
        freeze_teacher=config['model']['teacher']['freeze'],
        teacher_checkpoint=config['model']['teacher']['checkpoint_path'],
        p_encoder=config['model']['student']['p_encoder'],
        p_classifier=config['model']['student']['p_classifier']
    ).to(device)

    # DDP 包装
    model = DDP(model, device_ids=[local_gpu_id], find_unused_parameters=True)

    # 创建自定义损失函数（设置 lambda_m 和 beta）
    criterion = DistillationLoss(
        temperature=config['distillation']['temperature'],
        alpha=config['distillation']['alpha'],
        learnable_alpha=config['distillation'].get('learnable_alpha', False),
        weight_type=config['distillation'].get('weight_type', 'uniform'),
        distill_type=config['distillation'].get('distill_type', 'MTSKD_Temp'),
        seq_len=config['model'].get('stu_seq_len', 16),
        use_dynamic=config['distillation'].get('USE_DYNAMIC_DISTILL_WEIGHT', False),
        num_classes=config['model']['num_classes']
    ).to(device)

    # 🔧 设置固定权重（关键步骤）
    with torch.no_grad():
        criterion.mtskd_weight.data.fill_(lambda_m)
        criterion.beta.data.fill_(beta)

    # 优化器（仅优化学生网络参数）
    student_params = list(model.module.student.parameters())
    optimizer = optim.AdamW(
        student_params,
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )

    # 训练循环
    best_val_acc = 0.0

    if rank == 0:
        logger.info(f"{'='*60}")
        logger.info(f"开始训练: lambda_m={lambda_m}, beta={beta}")
        logger.info(f"训练轮数: {num_epochs}")
        logger.info(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        # 设置 epoch 以确保每个 epoch 的 shuffle 不同
        train_sampler.set_epoch(epoch)

        # ===== 通知 loss 当前 epoch（用于动态蒸馏）=====
        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(epoch, num_epochs)

        # 训练一个 epoch
        model.train()
        train_loss = 0.0
        total_hard_loss = 0.0
        total_soft_loss = 0.0
        correct = 0
        total = 0
        valid_batches = 0
        log_interval = config['training'].get('log_interval', 20)

        for batch_idx, (input_data, labels) in enumerate(train_loader):
            inputs = get_inputs(input_data, config['dataset']['data_type'], device)
            labels = labels.to(device)

            # 前向传播
            student_logits, stu_seq_logits, fl, fg, bl, bg, x_encoder, teacher_logits, output_cnn_features, teacher_all_hidden_states = model(inputs)

            # 计算损失
            loss, hard_loss, soft_loss, alpha, beta_val, memkd_weight = criterion(
                student_logits, stu_seq_logits, fl, fg, bl, bg, x_encoder,
                teacher_logits, output_cnn_features, labels, teacher_all_hidden_states
            )

            # 检查 loss 是否有异常值
            if torch.isnan(loss) or torch.isinf(loss):
                if rank == 0:
                    logger.warning(f'Invalid loss at epoch {epoch}, batch {batch_idx}')
                continue

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 统计
            train_loss += loss.item()
            total_hard_loss += hard_loss.item()
            total_soft_loss += soft_loss.item()
            pred = student_logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            valid_batches += 1

            # 训练过程中的日志输出（每个 log_interval 个 batch）
            if rank == 0 and batch_idx % log_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(
                    f'Epoch [{epoch}] Batch [{batch_idx}/{len(train_loader)}] '
                    f'Loss: {loss.item():.4f}, Hard: {hard_loss.item():.4f}, Soft: {soft_loss.item():.4f} '
                    f'Acc: {100.*correct/total:.2f}% LR: {current_lr:.6f} '
                    f'Alpha: {alpha:.4f} '
                    f'Beta: {beta_val:.4f} '
                    f'MemKD: {memkd_weight:.4f}'
                )

        train_loss_avg = train_loss / max(valid_batches, 1)
        hard_loss_avg = total_hard_loss / max(valid_batches, 1)
        soft_loss_avg = total_soft_loss / max(valid_batches, 1)
        train_acc = 100. * correct / max(total, 1)

        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for input_data, labels in val_loader:
                inputs = get_inputs(input_data, config['dataset']['data_type'], device)
                labels = labels.to(device)

                student_logits, _, _, _, _, _, _, teacher_logits, _, _ = model(inputs)

                loss = nn.functional.cross_entropy(student_logits, labels)
                val_loss += loss.item()

                pred = student_logits.argmax(dim=1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)

        # DDP 下聚合结果
        if dist.is_initialized():
            val_metrics_tensor = torch.tensor([val_loss, val_correct, val_total], device=device)
            dist.all_reduce(val_metrics_tensor, op=dist.ReduceOp.SUM)
            val_loss, val_correct, val_total = val_metrics_tensor.tolist()

        num_val_batches = len(val_loader)
        if dist.is_initialized():
            num_val_batches *= dist.get_world_size()

        val_acc = 100. * val_correct / max(val_total, 1)
        val_loss_avg = val_loss / max(num_val_batches, 1)

        # 更新最佳验证准确率
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        # 🔥 保存最佳模型（如果启用）
        if save_model and rank == 0 and is_best and save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            # 保存学生模型
            if isinstance(model, DDP):
                student_state = model.module.student.state_dict()
            else:
                student_state = model.student.state_dict()
            
            model_path = os.path.join(
                save_dir, 
                f'best_model_lambda_m_{lambda_m:.2f}_beta_{beta:.2f}.pth'
            )
            torch.save({
                'student_state_dict': student_state,
                'val_acc': val_acc,
                'lambda_m': lambda_m,
                'beta': beta,
                'epoch': epoch
            }, model_path)
            logger.info(f'💾 保存最佳模型: {model_path} (Val Acc: {val_acc:.2f}%)')

        # 每个 epoch 都输出日志
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Epoch [{epoch}/{num_epochs}] '
                f'Train Loss: {train_loss_avg:.4f}, Hard: {hard_loss_avg:.4f}, Soft: {soft_loss_avg:.4f}, '
                f'Train Acc: {train_acc:.2f}%, '
                f'Val Loss: {val_loss_avg:.4f}, '
                f'Val Acc: {val_acc:.2f}%, '
                f'Best Val Acc: {best_val_acc:.2f}%, '
                f'LR: {current_lr:.6f}'
            )

        # 学习率调度
        scheduler.step()

    if rank == 0:
        logger.info(f"✅ 训练完成! lambda_m={lambda_m}, beta={beta}, 最佳验证准确率: {best_val_acc:.2f}%")
        logger.info(f"{'='*60}\n")

    # 同步最佳准确率到所有进程
    if dist.is_initialized():
        best_acc_tensor = torch.tensor(best_val_acc, device=device)
        dist.all_reduce(best_acc_tensor, op=dist.ReduceOp.MAX)
        best_val_acc = best_acc_tensor.item()

    return best_val_acc


# ============================================
# 超参数扫描函数
# ============================================

def step_a_scan_lambda_m(
    config: dict,
    rank: int,
    logger,
    beta: float = 0.5,
    lambda_m_values: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    num_epochs: int = 20
) -> Tuple[List[Dict], float]:
    """
    Step A: 扫描 lambda_m（固定 beta）

    输入：
        config: dict - 配置字典
        rank: int - 进程rank
        logger: 日志记录器
        beta: float - 固定的 beta 值
        lambda_m_values: List[float] - 要扫描的 lambda_m 值列表
        num_epochs: int - 每次实验的训练轮数

    输出：
        results: List[Dict] - 包含所有实验结果的列表
        best_lambda_m: float - 最佳的 lambda_m 值
    """
    if rank == 0:
        logger.info(f"\n{'#'*60}")
        logger.info(f"Step A: 扫描 lambda_m (固定 beta={beta})")
        logger.info(f"{'#'*60}\n")

    results = []

    for lambda_m in lambda_m_values:
        # 运行训练
        val_acc = train_and_evaluate(
            lambda_m=lambda_m,
            beta=beta,
            config=config,
            rank=rank,
            logger=logger,
            num_epochs=num_epochs
        )

        # 保存结果
        result = {
            'lambda_m': lambda_m,
            'beta': beta,
            'val_acc': val_acc
        }
        results.append(result)

        # 打印结果（仅主进程）
        if rank == 0:
            logger.info(f"lambda_m={lambda_m:.2f}, beta={beta:.2f} -> Val Acc: {val_acc:.2f}%")

    # 找到最佳 lambda_m（仅主进程）
    if rank == 0:
        best_result = max(results, key=lambda x: x['val_acc'])
        best_lambda_m = best_result['lambda_m']
        best_acc = best_result['val_acc']
        logger.info(f"\n{'='*60}")
        logger.info(f"Step A 完成! 最佳 lambda_m = {best_lambda_m:.2f} (Val Acc: {best_acc:.2f}%)")
        logger.info(f"{'='*60}\n")
    else:
        best_lambda_m = None

    # 同步最佳 lambda_m 到所有进程
    if dist.is_initialized():
        gpu_id = config['distributed']['gpu_ids'][rank]
        best_lambda_m_tensor = torch.tensor(
            best_lambda_m if best_lambda_m is not None else 0.0,
            device=torch.device(f'cuda:{gpu_id}')
        )
        dist.broadcast(best_lambda_m_tensor, src=0)
        best_lambda_m = best_lambda_m_tensor.item()

    return results, best_lambda_m


def step_b_scan_beta(
    config: dict,
    rank: int,
    logger,
    lambda_m: float,
    beta_values: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    num_epochs: int = 20
) -> List[Dict]:
    """
    Step B: 扫描 beta（固定 lambda_m）

    输入：
        config: dict - 配置字典
        rank: int - 进程rank
        logger: 日志记录器
        lambda_m: float - 固定的 lambda_m 值（最佳值）
        beta_values: List[float] - 要扫描的 beta 值列表
        num_epochs: int - 每次实验的训练轮数

    输出：
        results: List[Dict] - 包含所有实验结果的列表
    """
    if rank == 0:
        logger.info(f"\n{'#'*60}")
        logger.info(f"Step B: 扫描 beta (固定 lambda_m={lambda_m})")
        logger.info(f"{'#'*60}\n")

    results = []

    for beta in beta_values:
        # 运行训练
        val_acc = train_and_evaluate(
            lambda_m=lambda_m,
            beta=beta,
            config=config,
            rank=rank,
            logger=logger,
            num_epochs=num_epochs
        )

        # 保存结果
        result = {
            'lambda_m': lambda_m,
            'beta': beta,
            'val_acc': val_acc
        }
        results.append(result)

        # 打印结果（仅主进程）
        if rank == 0:
            logger.info(f"lambda_m={lambda_m:.2f}, beta={beta:.2f} -> Val Acc: {val_acc:.2f}%")

    # 找到最佳 beta（仅主进程）
    if rank == 0:
        best_result = max(results, key=lambda x: x['val_acc'])
        best_beta = best_result['beta']
        best_acc = best_result['val_acc']
        logger.info(f"\n{'='*60}")
        logger.info(f"Step B 完成! 最佳 beta = {best_beta:.2f} (Val Acc: {best_acc:.2f}%)")
        logger.info(f"{'='*60}\n")

    return results


# ============================================
# 结果保存函数
# ============================================

def save_results_to_csv(results: List[Dict], save_path: str, rank: int):
    """
    保存结果到 CSV 文件

    输入：
        results: List[Dict] - 实验结果列表
        save_path: str - 保存路径
        rank: int - 进程rank（只有rank=0会保存）
    """
    if rank != 0:
        return

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['lambda_m', 'beta', 'val_acc'])
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ 结果已保存到: {save_path}")


def save_summary_to_csv(
    step_a_results: List[Dict],
    step_b_results: List[Dict],
    best_lambda_m: float,
    save_path: str,
    rank: int
):
    """
    保存汇总结果到 CSV 文件（包含两步的最佳配置）

    输入：
        step_a_results: List[Dict] - Step A 的结果
        step_b_results: List[Dict] - Step B 的结果
        best_lambda_m: float - 最佳 lambda_m 值
        save_path: str - 保存路径
        rank: int - 进程rank
    """
    if rank != 0:
        return

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Step', 'lambda_m', 'beta', 'val_acc', 'is_best'])

        # 写入 Step A 结果
        best_val_acc_a = max([r['val_acc'] for r in step_a_results])
        for result in step_a_results:
            is_best = result['val_acc'] == best_val_acc_a
            writer.writerow([
                'Step_A',
                result['lambda_m'],
                result['beta'],
                result['val_acc'],
                is_best
            ])

        # 写入 Step B 结果
        best_val_acc_b = max([r['val_acc'] for r in step_b_results])
        for result in step_b_results:
            is_best = result['val_acc'] == best_val_acc_b
            writer.writerow([
                'Step_B',
                result['lambda_m'],
                result['beta'],
                result['val_acc'],
                is_best
            ])

    print(f"✅ 汇总结果已保存到: {save_path}")


# ============================================
# 主函数
# ============================================

def main_worker(rank: int, world_size: int, config: dict, gpu_ids: list, args):
    """单个 GPU 的工作进程"""
    # 设置 DDP
    setup_ddp(rank, world_size, config, gpu_ids)

    # 创建保存目录
    save_dir = config['save']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    # 设置日志
    logger = setup_logger(save_dir, rank)

    if rank == 0:
        logger.info("="*60)
        logger.info("超参数分析开始")
        logger.info("="*60)
        logger.info(f"配置: {args.config}")
        logger.info(f"GPUs: {gpu_ids}")
        logger.info(f"World size: {world_size}")
        logger.info(f"Step A - 扫描 lambda_m: {args.lambda_m_values}")
        logger.info(f"Step B - 扫描 beta: {args.beta_values}")
        logger.info(f"每轮训练 epochs: {args.num_epochs}")
        logger.info(f"固定 beta (Step A): {args.fixed_beta}")
        logger.info("="*60)

    # 超参数取值
    lambda_m_values = [float(x) for x in args.lambda_m_values.split(',')]
    beta_values = [float(x) for x in args.beta_values.split(',')]
    fixed_beta = args.fixed_beta

    # ============================================
    # Step A: 扫描 lambda_m
    # ============================================
    step_a_results, best_lambda_m = step_a_scan_lambda_m(
        config=config,
        rank=rank,
        logger=logger,
        beta=fixed_beta,
        lambda_m_values=lambda_m_values,
        num_epochs=args.num_epochs
    )

    # 保存 Step A 结果
    step_a_csv_path = os.path.join(save_dir, 'step_a_lambda_m_scan.csv')
    save_results_to_csv(step_a_results, step_a_csv_path, rank)

    # ============================================
    # Step B: 扫描 beta
    # ============================================
    step_b_results = step_b_scan_beta(
        config=config,
        rank=rank,
        logger=logger,
        lambda_m=best_lambda_m,
        beta_values=beta_values,
        num_epochs=args.num_epochs
    )

    # 保存 Step B 结果
    step_b_csv_path = os.path.join(save_dir, 'step_b_beta_scan.csv')
    save_results_to_csv(step_b_results, step_b_csv_path, rank)

    # ============================================
    # 保存汇总结果
    # ============================================
    summary_csv_path = os.path.join(save_dir, 'hyperparameter_tuning_summary.csv')
    save_summary_to_csv(step_a_results, step_b_results, best_lambda_m, summary_csv_path, rank)

    if rank == 0:
        logger.info("\n" + "="*60)
        logger.info("超参数分析完成!")
        logger.info("="*60)
        logger.info(f"最佳配置: lambda_m={best_lambda_m:.2f}, beta={max(step_b_results, key=lambda x: x['val_acc'])['beta']:.2f}")
        logger.info(f"最高准确率: {max(step_b_results, key=lambda x: x['val_acc'])['val_acc']:.2f}%")
        logger.info("="*60)

    # 清理 DDP
    cleanup_ddp()


def parse_gpu_ids(gpu_str: str) -> list:
    """解析 GPU ID 字符串"""
    return [int(x.strip()) for x in gpu_str.split(',')]


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for Distillation Loss')
    parser.add_argument('--config', type=str,
                        default='/media/hdd1/fubohan/Code/UATR/configs/hyperparameter_tuning.yaml',
                        help='Path to config file')
    parser.add_argument('--gpus', type=str, default=None,
                        help='指定GPU，如 "0,1,2,3" 或 "2,3"')
    parser.add_argument('--lambda_m_values', type=str, default='0.0,0.25,0.5,0.75,1.0',
                        help='Step A 扫描的 lambda_m 值（逗号分隔）')
    parser.add_argument('--beta_values', type=str, default='0.0,0.25,0.5,0.75,1.0',
                        help='Step B 扫描的 beta 值（逗号分隔）')
    parser.add_argument('--fixed_beta', type=float, default=0.5,
                        help='Step A 中固定的 beta 值')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='每次训练的轮数（如果不指定，则使用 YAML 中的配置）')

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 确定使用的 epoch 数量
    # 如果命令行没有指定，则使用 YAML 中的配置
    if args.num_epochs is None:
        args.num_epochs = config['training']['num_epochs']
        print(f"使用 YAML 配置中的 epoch 数量: {args.num_epochs}")
    else:
        print(f"使用命令行指定的 epoch 数量: {args.num_epochs}")

    # 设置保存目录（添加超参数搜索的子目录）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config['save']['save_dir'] = os.path.join(
        config['save']['save_dir'],
        f'hyperparam_search_{timestamp}'
    )

    # 确定使用的 GPU
    if args.gpus is not None:
        gpu_ids = parse_gpu_ids(args.gpus)
    elif 'gpu_ids' in config['distributed'] and config['distributed']['gpu_ids']:
        gpu_ids = config['distributed']['gpu_ids']
    else:
        gpu_ids = list(range(torch.cuda.device_count()))

    # 验证 GPU ID 是否有效
    available_gpus = torch.cuda.device_count()
    for gpu_id in gpu_ids:
        if gpu_id >= available_gpus:
            raise ValueError(f'GPU {gpu_id} 不存在，可用GPU数量: {available_gpus}')

    world_size = len(gpu_ids)

    print(f'Using GPUs: {gpu_ids}')
    print(f'World size: {world_size}')
    print(f'Config loaded from: {args.config}')
    print(f'Save dir: {config["save"]["save_dir"]}')

    # 启动多进程训练
    torch.multiprocessing.spawn(
        main_worker,
        args=(world_size, config, gpu_ids, args),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()
