import os
import argparse
import yaml
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# from models.CFC import AudioCfC
from models.Audio_TeacherNet import build_Audio_TeacherNet
from utils.distillation_loss import DistillationLoss
from datasets.audio_dataset import AudioDataset, train_transform, validation_test_transform
from models.distillation import AudioDistillationModel
# from models.LNN import AudioCfC


def setup_logger(save_dir: str, rank: int) -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger('distillation')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # 清空已有 handlers
    
    # 只在主进程记录日志
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(save_dir, f'train_{timestamp}.log')
        
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


def save_checkpoint(model: nn.Module, optimizer, scheduler, epoch: int,
                    best_acc: float, save_path: str, is_best: bool = False):
    """保存检查点"""
    # 获取学生网络的 state_dict
    if isinstance(model, DDP):
        student_state = model.module.student.state_dict()
    else:
        student_state = model.student.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'student_state_dict': student_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), 'best_student.pth')
        torch.save({'student_state_dict': student_state, 'best_acc': best_acc}, best_path)

def get_inputs(input_data, data_type, device):
    """根据数据类型获取输入"""
    if '@' not in data_type:
        # 单一数据类型
        return input_data[0].to(device)
    else:
        # 多模态输入
        return [data.to(device) for data in input_data]


class DistillationTrainer:
    """蒸馏训练器 (支持 DDP)"""
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
        self.data_type = config['dataset']['data_type']
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> dict:
        """训练一个 epoch"""
        self.model.train()
        
        # 教师网络保持 eval 模式
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
            # 使用与 train.py 相同的方式获取输入
            inputs = get_inputs(input_data, self.data_type, self.device)
            labels = labels.to(self.device)
            
            # 检查输入是否有异常值
            # if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            #     self.logger.warning(f'NaN/Inf in inputs at epoch {epoch}, batch {batch_idx}')
            #     inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 前向传播
            student_logits, stu_seq_logits, teacher_logits = self.model(inputs)
            
            # 检查输出是否有异常值
            if torch.isnan(student_logits).any() or torch.isinf(student_logits).any():
                self.logger.warning(f'NaN/Inf in outputs at epoch {epoch}, batch {batch_idx}')
                continue
            
            # 计算损失
            loss, hard_loss, soft_loss, alpha = self.criterion(
                student_logits, stu_seq_logits, teacher_logits, labels
            )
            
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
            total_hard_loss += hard_loss.item()
            total_soft_loss += soft_loss.item()
            
            pred = student_logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            valid_batches += 1
            
            if self.rank == 0 and batch_idx % self.config['training']['log_interval'] == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(
                    f'Epoch [{epoch}] Batch [{batch_idx}/{len(train_loader)}] '
                    f'Loss: {loss.item():.4f}, Hard: {hard_loss.item():.4f}, Soft: {soft_loss.item():.4f} '
                    f'Acc: {100.*correct/total:.2f}% LR: {current_lr:.6f}'
                    f' Alpha: {alpha:.4f}'
                )
        
        metrics = {
            'loss': total_loss / max(valid_batches, 1),
            'hard_loss': total_hard_loss / max(valid_batches, 1),
            'soft_loss': total_soft_loss / max(valid_batches, 1),
            'accuracy': correct / max(total, 1)
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict:
        """验证"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for input_data, labels in val_loader:
            inputs = get_inputs(input_data, self.data_type, self.device)
            labels = labels.to(self.device)
            
            # 检查输入，单模态输入直接检查， 多模态输入检查每个模态，20260121yy修改
            # if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            #     inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1.0, neginf=-1.0)
            has_nan_inf = False
            if isinstance(inputs, list):
                # 多模态输入：检查列表中的每个张量
                for i, tensor in enumerate(inputs):
                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        has_nan_inf = True
                # 统一处理NaN/Inf
                if has_nan_inf:
                    inputs = [torch.nan_to_num(inp, nan=0.0, posinf=1.0, neginf=-1.0) for inp in inputs]
            else:
                # 单模态输入：直接检查
                if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1.0, neginf=-1.0)
            # 仅使用学生网络
            student_logits, _, _ = self.model(inputs)
            
            # 检查输出
            if torch.isnan(student_logits).any() or torch.isinf(student_logits).any():
                continue
            
            loss = F.cross_entropy(student_logits, labels)
            
            total_loss += loss.item()
            pred = student_logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        
        # DDP 下聚合结果
        if dist.is_initialized():
            metrics_tensor = torch.tensor([total_loss, correct, total], device=self.device)
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            total_loss, correct, total = metrics_tensor.tolist()
        
        num_batches = len(val_loader)
        if dist.is_initialized():
            num_batches *= dist.get_world_size()
        
        metrics = {
            'loss': total_loss / max(num_batches, 1),
            'accuracy': 100. * correct / max(total, 1)
        }
        
        return metrics
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        """完整训练流程"""
        num_epochs = self.config['training']['num_epochs']
        save_dir = self.config['save']['save_dir'] # 修改保存目录
        save_interval = self.config['save']['save_interval']
        
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(1, num_epochs + 1):
            # DDP: 设置 epoch 以确保每个 epoch 的 shuffle 不同
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_metrics = self.validate(val_loader)
            
            # 学习率调度
            self.scheduler.step()
            
            # 日志记录 (仅主进程)
            if self.rank == 0:
                self.logger.info(
                    f'Epoch [{epoch}/{num_epochs}] '
                    f'Train Loss: {train_metrics["loss"]:.4f}, '
                    f'Train Acc: {100.*train_metrics["accuracy"]:.2f}%, '
                    f'Val Loss: {val_metrics["loss"]:.4f}, '
                    f'Val Acc: {val_metrics["accuracy"]:.2f}%'
                )
                
                
                # 保存最佳模型
                is_best = val_metrics['accuracy'] > self.best_acc
                if is_best:
                    self.best_acc = val_metrics['accuracy']
                    self.logger.info(f'Best model saved with accuracy: {self.best_acc:.2f}%')
                
                # 定期保存或保存最佳
                if epoch % save_interval == 0 or is_best:
                    save_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
                    save_checkpoint(
                        self.model, self.optimizer, self.scheduler,
                        epoch, self.best_acc, save_path, is_best
                    )
                    if epoch % save_interval == 0:
                        self.logger.info(f'Checkpoint saved at epoch {epoch}')
        
        if self.rank == 0:
            self.logger.info('='*50)
            self.logger.info(f'Training finished! Best accuracy: {self.best_acc:.2f}%')
            self.logger.info('='*50)


def main_worker(rank: int, world_size: int, config: dict, gpu_ids: list):
    """单个 GPU 的训练进程"""
    # 设置 DDP
    setup_ddp(rank, world_size, config, gpu_ids)
    
    # 获取当前进程对应的实际 GPU ID
    local_gpu_id = gpu_ids[rank]
    device = torch.device(f'cuda:{local_gpu_id}')
    
    # 设置随机种子
    torch.manual_seed(config['training']['seed'] + rank)
    torch.cuda.manual_seed(config['training']['seed'] + rank)
    
    # 创建保存目录
    save_dir = config['save']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # 日志和 TensorBoard
    logger = setup_logger(save_dir, rank)
    if rank == 0:
        logger.info('='*50)
        logger.info('Distillation Training Configuration')
        logger.info('='*50)
        logger.info(f'GPUs: {gpu_ids}')
        logger.info(f'World size: {world_size}')
        logger.info(f'Dataset: {config["dataset"]["data_dir"]}')
        logger.info(f'Data type: {config["dataset"]["data_type"]}')
        logger.info(f'Batch size: {config["training"]["batch_size"]}')
        logger.info(f'Epochs: {config["training"]["num_epochs"]}')
        logger.info(f'Learning rate: {config["training"]["lr"]}')
        logger.info(f'Temperature: {config["distillation"]["temperature"]}')
        logger.info(f'Alpha: {config["distillation"]["alpha"]}')
        logger.info(f'Save dir: {save_dir}')
        logger.info('='*50)
    
    # 创建数据集 (使用与 train.py 相同的方式)
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
    debug_num_classes = config['model']['num_classes']
    print(f"[DEBUG] Creating AudioDistillationModel with num_classes={debug_num_classes}")
    model = AudioDistillationModel(
        num_classes=debug_num_classes,
        teacher_pretrained=config['model']['teacher']['pretrained'],
        freeze_teacher=config['model']['teacher']['freeze'],
        teacher_checkpoint=config['model']['teacher']['checkpoint_path']
        # LNN AudioCfC使用默认参数，无需传入额外参数
    ).to(device)
    
    if rank == 0:
        # 打印模型参数量
        student_params = sum(p.numel() for p in model.student.parameters())
        teacher_params = sum(p.numel() for p in model.teacher.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'Student params: {student_params:,} ({student_params/1e6:.2f}M)')
        logger.info(f'Teacher params: {teacher_params:,} ({teacher_params/1e6:.2f}M)')
        logger.info(f'Trainable params: {trainable_params:,} ({trainable_params/1e6:.2f}M)')
    
    # DDP 包装
    model = DDP(model, device_ids=[local_gpu_id], find_unused_parameters=True)
    
    # 损失函数
    criterion = DistillationLoss(
        temperature=config['distillation']['temperature'],
        alpha=config['distillation']['alpha'],
        learnable_alpha=config['distillation'].get('learnable_alpha', False),
        weight_type=config['distillation'].get('weight_type', 'uniform'),
        distill_type=config['distillation'].get('distill_type', 'kl'),
        # yaml写了就用yaml的值，没写就使用这里默认的值
    ).to(device)
    
    # 优化器 (仅优化学生网络和可学习的 alpha)
    student_params = list(model.module.student.parameters())
    params_to_optimize = student_params + list(criterion.parameters())
    
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
    trainer = DistillationTrainer(
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
        logger.info('Start training...')
    
    # 开始训练
    trainer.fit(train_loader, val_loader)
    
    # 清理

    cleanup_ddp()


def parse_gpu_ids(gpu_str: str) -> list:
    """解析 GPU ID 字符串"""
    return [int(x.strip()) for x in gpu_str.split(',')]


def main():
    parser = argparse.ArgumentParser(description='Audio Distillation Training')
    parser.add_argument('--config', type=str,
                        default='/media/hdd1/fubohan/Code/UATR/configs/train_distillation_deepship.yaml',
                        help='Path to config file')
    parser.add_argument('--gpus', type=str, default=None,
                        help='指定GPU，如 "0,1,2,3" 或 "2,3"')
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
    
    # 验证 GPU ID 是否有效
    available_gpus = torch.cuda.device_count()
    for gpu_id in gpu_ids:
        if gpu_id >= available_gpus:
            raise ValueError(f'GPU {gpu_id} 不存在，可用GPU数量: {available_gpus}')
    
    world_size = len(gpu_ids)
    
    # 创建保存目录
    os.makedirs(config['save']['save_dir'], exist_ok=True)
    
    print(f'Using GPUs: {gpu_ids}')
    print(f'World size: {world_size}')
    print(f'Config loaded from: {args.config}')
    
    # 启动多进程训练
    torch.multiprocessing.spawn(
        main_worker,
        args=(world_size, config, gpu_ids),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()