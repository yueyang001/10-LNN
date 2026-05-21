import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import os  # 关键修复：添加os模块导入
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, RandomSampler
 
# 引入您的模型和数据集定义
from models.distillation import AudioDistillationModel
from datasets.audio_dataset import AudioDataset, validation_test_transform
 
def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
 
def get_inputs(input_data, data_type, device):
    if '@' not in data_type:
        return input_data[0].to(device)
    else:
        return [data.to(device) for data in input_data]
 
def plot_tsne(features, labels, title, save_path, num_classes):
    """绘制 t-SNE 图 (彻底修复版)"""
    print(f"📉 Running t-SNE for {title}...")
    print(f"   - Total samples: {len(features)}")
    print(f"   - Unique classes found: {np.unique(labels)}")
    
    # 兼容不同版本scikit-learn的写法
    # 移除所有可能导致兼容问题的参数，只保留最核心的
    tsne = TSNE(
        n_components=2, 
        random_state=42, 
        perplexity=30,
        init='pca',  # 添加init参数提升稳定性
        learning_rate='auto'  # 自动学习率，适配新版sklearn
    )
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 10))
    
    # 设置颜色映射：tab10 适合 10 类以内，非常鲜艳
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # 为每个类别单独画点，方便控制图例
    for i in range(num_classes):
        idx = labels == i
        if np.sum(idx) > 0:
            plt.scatter(
                features_2d[idx, 0], 
                features_2d[idx, 1], 
                c=[colors[i]], 
                label=f'Class {i}',
                s=15, 
                alpha=0.8,
                edgecolors='k',
                linewidths=0.2
            )
    
    plt.title(title, fontsize=16)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved colorful plot: {save_path}")
 
def plot_dynamic_trajectory(student_seq, teacher_seq, title, save_path):
    student_norms = np.linalg.norm(student_seq, axis=1)
    teacher_norms = np.linalg.norm(teacher_seq, axis=1)
    
    if len(teacher_norms) != len(student_norms):
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, len(teacher_norms))
        x_new = np.linspace(0, 1, len(student_norms))
        f = interp1d(x_old, teacher_norms, kind='linear')
        teacher_norms_aligned = f(x_new)
    else:
        teacher_norms_aligned = teacher_norms
 
    plt.figure(figsize=(10, 6))
    plt.plot(student_norms, label='Student Trajectory (LNN)', color='#1f77b4', marker='o', markersize=4, linewidth=2)
    plt.plot(teacher_norms_aligned, label='Teacher Trajectory (CNN, Aligned)', color='#d62728', linestyle='--', marker='x', markersize=4, linewidth=2)
    
    plt.title(title)
    plt.xlabel("Time Step (Normalized)")
    plt.ylabel("Feature L2 Norm")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")
 
def main():
    parser = argparse.ArgumentParser(description='Experiment 6: Feature Visualization')
    
    # === DeepShip 最佳模型参数 ===
    parser.add_argument('--config', type=str, 
                        default='/media/hdd1/fubohan/Code/UATR/checkpoints/grid_deepship_0/num_epochs200_weight_decay8e-05_batch_size16_lr0.0004_distill_typeMTSKD_Temp_freezeTrue_USE_DYNAMIC_DISTILL_WEIGHTFalse_p_encoder0.1_p_classifier0.35/config.yaml')
    parser.add_argument('--checkpoint', type=str, 
                        default='/media/hdd1/fubohan/Code/UATR/checkpoints/grid_deepship_0/num_epochs200_weight_decay8e-05_batch_size16_lr0.0004_distill_typeMTSKD_Temp_freezeTrue_USE_DYNAMIC_DISTILL_WEIGHTFalse_p_encoder0.1_p_classifier0.35/best_student.pth')
    parser.add_argument('--output_dir', type=str, default='results/experiment6_deepship')
    parser.add_argument('--max_samples', type=int, default=4000, help='最大采样数量')
    
    args = parser.parse_args()
 
    setup_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
 
    # 1. 加载配置
    print(f"📂 Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    num_classes = config['model']['num_classes']
    
    # 2. 加载模型
    print("📦 Building model and loading weights...")
    model = AudioDistillationModel(
        num_classes=num_classes,
        teacher_pretrained=config['model']['teacher']['pretrained'],
        freeze_teacher=config['model']['teacher']['freeze'],
        teacher_checkpoint=config['model']['teacher']['checkpoint_path'],
        p_encoder=config['model']['student']['p_encoder'],
        p_classifier=config['model']['student']['p_classifier']
    ).to(device)
 
    ckpt = torch.load(args.checkpoint, map_location=device)
    if 'student_state_dict' in ckpt:
        model.student.load_state_dict(ckpt['student_state_dict'])
    else:
        model.student.load_state_dict(ckpt, strict=False)
 
    model.eval()
 
    # 3. 加载数据
    print("📂 Loading validation data with shuffle=True...")
    val_dataset = AudioDataset(
        data_dir=config['dataset']['data_dir'],
        data_flag='validation',
        data_type=config['dataset']['data_type'],
        transform=validation_test_transform
    )
    
    sampler = RandomSampler(val_dataset, replacement=False)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        sampler=sampler, 
        num_workers=4
    )
 
    # 4. 提取特征
    all_student_features = []
    all_labels = []
    
    saved_student_seq = None
    saved_teacher_seq = None
 
    print("🚀 Extracting features...")
    with torch.no_grad():
        for batch_idx, (input_data, labels) in enumerate(val_loader):
            inputs = get_inputs(input_data, config['dataset']['data_type'], device)
            
            outputs = model(inputs)
            
            student_logits = outputs[0]
            x_encoder = outputs[6]
            output_cnn_features = outputs[8]
            
            all_student_features.append(student_logits.cpu().numpy())
            all_labels.append(labels.numpy())
            
            if saved_student_seq is None:
                saved_student_seq = x_encoder[0].cpu().numpy()
                saved_teacher_seq = output_cnn_features[0].cpu().numpy()
 
            current_count = (batch_idx + 1) * val_loader.batch_size  # 修复计数逻辑
            unique_labels = np.unique(np.concatenate(all_labels))
            
            if current_count >= args.max_samples and len(unique_labels) == num_classes:
                print(f"   Collected {current_count} samples with classes {unique_labels}. Stopping extraction.")
                break
 
    # 5. 生成可视化
    all_student_features = np.concatenate(all_student_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    plot_tsne(
        features=all_student_features,
        labels=all_labels,
        title="DeepShip: Student Feature Distribution (t-SNE)",
        save_path=os.path.join(args.output_dir, "fig1_tsne_deepship_colorful.png"),
        num_classes=num_classes
    )
 
    if saved_student_seq is not None and saved_teacher_seq is not None:
        plot_dynamic_trajectory(
            student_seq=saved_student_seq,
            teacher_seq=saved_teacher_seq,
            title="DeepShip: Dynamic Trajectory Alignment",
            save_path=os.path.join(args.output_dir, "fig2_trajectory_deepship.png")
        )
 
    print(f"\n✨ DeepShip Visualization Complete!")
 
if __name__ == '__main__':
    main()