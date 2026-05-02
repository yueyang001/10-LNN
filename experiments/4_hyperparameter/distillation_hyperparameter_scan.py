import copy
import logging
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict

import torch
from torch.utils.data import DataLoader

# Ensure project root is on sys.path when running from experiments folder
import sys
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from datasets.audio_dataset import AudioDataset, train_transform, validation_test_transform
from models.distillation import AudioDistillationModel
from utils.distillation_loss import DistillationLoss
from train_distillation_deepship import DistillationTrainer


def set_random_seed(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloaders(
    data_dir: str,
    data_type: str,
    batch_size: int,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = AudioDataset(
        data_dir=data_dir,
        data_flag='train',
        data_type=data_type,
        transform=train_transform,
    )
    val_dataset = AudioDataset(
        data_dir=data_dir,
        data_flag='validation',
        data_type=data_type,
        transform=validation_test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def make_criterion(lambda_m: float, beta: float, device: torch.device) -> DistillationLoss:
    criterion = DistillationLoss(
        temperature=4.0,
        alpha=0.5,
        learnable_alpha=False,
        weight_type='uniform',
        distill_type='MTSKD_Temp',
        seq_len=16,
        use_dynamic=False,
    ).to(device)

    # 设置 TS-T 中的 beta 权重
    criterion.beta.data = torch.tensor(beta, device=device, dtype=torch.float32)

    # 设置 lambda_m 权重为 MemKD vs TS-T
    lambda_m_clamped = float(np.clip(lambda_m, 1e-4, 1.0 - 1e-4))
    logit_value = np.log(lambda_m_clamped / (1.0 - lambda_m_clamped))
    criterion.mtskd_weight.data = torch.tensor(logit_value, device=device, dtype=torch.float32)

    return criterion


def build_model(
    num_classes: int,
    teacher_checkpoint: str,
    device: torch.device,
) -> AudioDistillationModel:
    model = AudioDistillationModel(
        num_classes=num_classes,
        teacher_pretrained=True,
        freeze_teacher=True,
        teacher_checkpoint=teacher_checkpoint,
        p_encoder=0.2,
        p_classifier=0.3,
    ).to(device)
    return model


def train_and_evaluate(
    lambda_m: float,
    beta: float,
    config: Dict,
) -> float:
    """
    输入：
        lambda_m: float
        beta: float
    输出：
        val_acc: float
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_random_seed(config['seed'])

    train_loader, val_loader = create_dataloaders(
        data_dir=config['data_dir'],
        data_type=config['data_type'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )

    model = build_model(
        num_classes=config['num_classes'],
        teacher_checkpoint=config['teacher_checkpoint'],
        device=device,
    )

    criterion = make_criterion(lambda_m=lambda_m, beta=beta, device=device)

    optimizer = torch.optim.AdamW(
        list(model.student.parameters()) + list(criterion.parameters()),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs']
    )

    config_run = copy.deepcopy(config)
    config_run['save'] = config_run.get('save', {}).copy()
    config_run['save']['save_dir'] = os.path.join(
        config_run['save'].get('save_dir', './hyperparameter_scan_checkpoints'),
        f'lambda_{lambda_m:.2f}_beta_{beta:.2f}'
    )

    logger = logging.getLogger(f'hyperparameter_scan_lambda_{lambda_m:.2f}_beta_{beta:.2f}')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    trainer = DistillationTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config_run,
        logger=logger,
        rank=0,
    )

    # 训练完整流程
    trainer.fit(train_loader, val_loader)

    # 返回验证准确率
    return trainer.best_acc


def scan_parameter(
    values: List[float],
    fixed: float,
    mode: str,
    config: Dict,
) -> List[Dict[str, float]]:
    results = []
    for value in values:
        if mode == 'lambda_m':
            print(f'>> 正在扫描 lambda_m={value:.2f}，beta={fixed:.2f}')
            val_acc = train_and_evaluate(lambda_m=value, beta=fixed, config=config)
            results.append({'lambda_m': value, 'beta': fixed, 'val_acc': val_acc})
            print(f'   lambda_m={value:.2f} -> Val Acc={val_acc:.4f}%')
        elif mode == 'beta':
            print(f'>> 正在扫描 beta={value:.2f}，lambda_m={fixed:.2f}')
            val_acc = train_and_evaluate(lambda_m=fixed, beta=value, config=config)
            results.append({'lambda_m': fixed, 'beta': value, 'val_acc': val_acc})
            print(f'   beta={value:.2f} -> Val Acc={val_acc:.4f}%')
        else:
            raise ValueError(f'Unknown scan mode: {mode}')

    return results


def save_csv(results: List[Dict[str, float]], path: str) -> None:
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    print(f'Saved results to {path}')


def plot_curve(x: List[float], y: List[float], xlabel: str, ylabel: str, title: str, save_path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker='o', linestyle='-', color='tab:blue')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f'Saved plot to {save_path}')


def main() -> None:
    config = {
        'seed': 2026,
        'data_dir': '/media/hdd1/fubohan/Code/UATR/datasets',
        'data_type': 'wav_s@wav_t',
        'batch_size': 8,
        'num_workers': 2,
        'num_epochs': 2,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'num_classes': 4,
        'teacher_checkpoint': '/media/hdd1/fubohan/Code/UATR/models/Audio_Teacher_ShipsEar_622/checkpoints/Student.pth',
        'dataset': {
            'data_type': 'wav_s@wav_t',
        },
        'training': {
            'num_epochs': 2,
            'log_interval': 20,
        },
        'model': {
            'teacher': {
                'freeze': True,
            }
        },
        'save': {
            'save_dir': './hyperparameter_scan_checkpoints',
            'save_interval': 1,
        },
    }

    lambda_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    beta_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    os.makedirs('results', exist_ok=True)

    print('=== Step A: Scan lambda_m (beta=0.5) ===')
    lambda_results = scan_parameter(lambda_values, fixed=0.5, mode='lambda_m', config=config)
    save_csv(lambda_results, 'results/lambda_m_results.csv')

    best_lambda_row = max(lambda_results, key=lambda row: row['val_acc'])
    best_lambda_m = best_lambda_row['lambda_m']
    print(f'Best lambda_m found: {best_lambda_m:.2f} with val_acc={best_lambda_row["val_acc"]:.4f}%')

    print('\n=== Step B: Scan beta (lambda_m={:.2f}) ==='.format(best_lambda_m))
    beta_results = scan_parameter(beta_values, fixed=best_lambda_m, mode='beta', config=config)
    save_csv(beta_results, 'results/beta_results.csv')

    plot_curve(
        x=[row['lambda_m'] for row in lambda_results],
        y=[row['val_acc'] for row in lambda_results],
        xlabel='lambda_m',
        ylabel='Validation Accuracy (%)',
        title='lambda_m vs Accuracy',
        save_path='results/lambda_m_vs_accuracy.png',
    )

    plot_curve(
        x=[row['beta'] for row in beta_results],
        y=[row['val_acc'] for row in beta_results],
        xlabel='beta',
        ylabel='Validation Accuracy (%)',
        title='beta vs Accuracy',
        save_path='results/beta_vs_accuracy.png',
    )

    print('\n=== Scan complete ===')
    print('lambda_m results:')
    for row in lambda_results:
        print(f"  lambda_m={row['lambda_m']:.2f} -> val_acc={row['val_acc']:.4f}%")
    print('beta results:')
    for row in beta_results:
        print(f"  beta={row['beta']:.2f} -> val_acc={row['val_acc']:.4f}%")


if __name__ == '__main__':
    main()
