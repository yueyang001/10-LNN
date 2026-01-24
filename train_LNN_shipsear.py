import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import logging
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .datasets.audio_dataset_old import AudioDataset, train_transform, validation_test_transform
# from models.LNN_Audio import liquidaudio_nano
from models.LNN import AudioCfC
# from models.CFC import AudioCfC as CFC_AudioCfC
# from models.LIQT2LNN import liquidnet_audio_tiny
# from models.LNN_Audio_encoder import AudioClassifier


def setup_logger(save_dir, rank):
    """设置日志记录器"""
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    
    # 只在主进程记录日志
    if rank == 0:
        # 创建日志文件名（包含时间戳）
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
        # 非主进程使用空处理器
        logger.addHandler(logging.NullHandler())
    
    return logger


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 确保数值类型正确
    config['train']['lr'] = float(config['train']['lr'])
    config['train']['weight_decay'] = float(config['train']['weight_decay'])
    
    return config


def setup(rank, world_size, config, gpu_ids):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = config['distributed']['master_addr']
    os.environ['MASTER_PORT'] = config['distributed']['master_port']
    dist.init_process_group(
        backend=config['distributed']['backend'],
        rank=rank,
        world_size=world_size
    )
    # 设置当前进程使用的GPU
    local_gpu_id = gpu_ids[rank]
    torch.cuda.set_device(local_gpu_id)


def cleanup():
    """清理分布式训练环境"""
    dist.destroy_process_group()


def train(rank, world_size, config, gpu_ids):
    # 初始化分布式环境
    setup(rank, world_size, config, gpu_ids)
    
    # 获取当前进程对应的实际GPU ID
    local_gpu_id = gpu_ids[rank]
    device = torch.device(f'cuda:{local_gpu_id}')
    
    # 创建保存目录
    save_dir = config['save']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置日志记录器
    logger = setup_logger(save_dir, rank)
    
    # 记录配置信息
    if rank == 0:
        logger.info('='*50)
        logger.info('Training Configuration')
        logger.info('='*50)
        logger.info(f'GPUs: {gpu_ids}')
        logger.info(f'World size: {world_size}')
        logger.info(f'Dataset: {config["dataset"]["data_dir"]}')
        logger.info(f'Data type: {config["dataset"]["data_type"]}')
        logger.info(f'Batch size: {config["train"]["batch_size"]}')
        logger.info(f'Epochs: {config["train"]["epochs"]}')
        logger.info(f'Learning rate: {config["train"]["lr"]}')
        logger.info(f'Weight decay: {config["train"]["weight_decay"]}')
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
    
    # 分布式采样器
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        sampler=train_sampler,
        num_workers=config['dataset']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        sampler=val_sampler,
        num_workers=config['dataset']['num_workers'],
        pin_memory=True,
        drop_last=False
    )
    
    # 初始化模型
    num_classes = config['model']['num_classes']
    model = AudioCfC(num_classes=num_classes)
    model = model.to(device)
    model = DDP(model, device_ids=[local_gpu_id], find_unused_parameters=True)
    
    # 记录模型信息
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'Total parameters: {total_params / 1e6:.2f}M')
        logger.info(f'Trainable parameters: {trainable_params / 1e6:.2f}M')
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay']
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['train']['step_size'],
        gamma=config['train']['gamma']
    )
    
    data_type = config['dataset']['data_type']
    
    # 训练循环
    best_acc = 0.0
    logger.info('Start training...')
    
    for epoch in range(config['train']['epochs']):
        model.train()
        train_sampler.set_epoch(epoch)
        
        total_loss = 0.0
        correct = 0
        total = 0
        valid_batches = 0
        
        for batch_idx, (input_data, labels) in enumerate(train_loader):
            inputs = get_inputs(input_data, data_type, device)
            labels = labels.to(device)
            
            # 检查输入是否有异常值
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                logger.warning(f'NaN/Inf in inputs at epoch {epoch}, batch {batch_idx}')
                inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1.0, neginf=-1.0)
            
            optimizer.zero_grad()
            outputs, sequence_output = model(inputs)
            
            # 检查输出是否有异常值
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                logger.warning(f'NaN/Inf in outputs at epoch {epoch}, batch {batch_idx}')
                continue
            # 在第217行之前添加标签检查
            # print(f"Labels range: {labels.min().item()} to {labels.max().item()}")
            # print(f"Expected range: 0 to {config['model']['num_classes'] - 1}")
            # unique_labels = torch.unique(labels)
            # print(f"Unique labels: {unique_labels}")
            loss = criterion(outputs, labels)
            
            # 检查loss是否有异常值
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f'Invalid loss at epoch {epoch}, batch {batch_idx}')
                continue
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            valid_batches += 1
            
            if rank == 0 and batch_idx % config['train']['log_interval'] == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f'Epoch [{epoch+1}/{config["train"]["epochs"]}] '
                           f'Batch [{batch_idx}/{len(train_loader)}] '
                           f'Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}% '
                           f'LR: {current_lr:.6f}')
        
        scheduler.step()
        
        # 验证
        val_acc = validate(model, val_loader, device, data_type, logger)
        
        if rank == 0:
            avg_loss = total_loss / max(valid_batches, 1)
            train_acc = 100. * correct / max(total, 1)
            logger.info(f'Epoch [{epoch+1}/{config["train"]["epochs"]}] '
                       f'Train Loss: {avg_loss:.4f} '
                       f'Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}%')
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(
                    model, optimizer, epoch, best_acc,
                    os.path.join(save_dir, 'best_model.pth')
                )
                logger.info(f'Best model saved with accuracy: {best_acc:.2f}%')
            
            # 定期保存模型 每10个epoch保存一次
            if (epoch + 1) % config['save']['save_interval'] == 0:
                save_checkpoint(
                    model, optimizer, epoch, val_acc,
                    os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                )
                logger.info(f'Checkpoint saved at epoch {epoch+1}')
    
    if rank == 0:
        logger.info('='*50)
        logger.info(f'Training finished! Best accuracy: {best_acc:.2f}%')
        logger.info('='*50)
    
    cleanup()


def get_inputs(input_data, data_type, device):
    """根据数据类型获取输入"""
    if '@' not in data_type:
        # 单一数据类型
        return input_data[0].to(device)
    else:
        # 多模态输入
        return [data.to(device) for data in input_data]


def validate(model, val_loader, device, data_type, logger=None):
    """验证函数"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for input_data, labels in val_loader:
            inputs = get_inputs(input_data, data_type, device)
            labels = labels.to(device)
            
            # 检查输入
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1.0, neginf=-1.0)
            
            outputs, sequence_output = model(inputs)
            
            # 检查输出
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                continue
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    # 聚合所有进程的结果
    correct_tensor = torch.tensor([correct], device=device)
    total_tensor = torch.tensor([total], device=device)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    
    accuracy = 100. * correct_tensor.item() / max(total_tensor.item(), 1)
    return accuracy


def save_checkpoint(model, optimizer, epoch, acc, path):
    """保存模型检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': acc,
    }, path)
    print(f"📁 Model saved to: {path}")


def parse_gpu_ids(gpu_str):
    """解析GPU ID字符串"""
    return [int(x.strip()) for x in gpu_str.split(',')]


def main():
    parser = argparse.ArgumentParser(description='DDP Training for Audio Classification')
    parser.add_argument('--config', type=str, default='configs/train_LNN_shipsear.yaml',
                        help='配置文件路径')
    parser.add_argument('--gpus', type=str, default=None,
                        help='指定GPU，如 "0,1,2,3" 或 "2,3"')
    args = parser.parse_args()
    
    # 加载配置
    import os
    config_path = args.config if os.path.isabs(args.config) else os.path.join(os.path.dirname(__file__), args.config)
    config = load_config(config_path)
    
    # 确定使用的GPU
    if args.gpus is not None:
        gpu_ids = parse_gpu_ids(args.gpus)
    elif 'gpu_ids' in config['distributed'] and config['distributed']['gpu_ids']:
        gpu_ids = config['distributed']['gpu_ids']
    else:
        gpu_ids = list(range(torch.cuda.device_count()))
    
    # 验证GPU ID是否有效
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
        train,
        args=(world_size, config, gpu_ids),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()