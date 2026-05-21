import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, RandomSampler
 
# 引入您的模型和数据集定义
from models.distillation import AudioDistillationModel
from datasets.audio_dataset import AudioDataset, validation_test_transform
 
def setup_seed(seed):
    """设置随机种子，保证实验可复现"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
 
def get_inputs(input_data, data_type, device):
    """适配不同数据类型的输入处理"""
    if '@' not in data_type:
        return input_data[0].to(device)
    else:
        return [data.to(device) for data in input_data]
 
def plot_tsne(features, labels, title, save_path, num_classes):
    """绘制 t-SNE 图 (适配5分类的shipsear数据集)"""
    print(f"📉 Running t-SNE for {title}...")
    print(f"   - Total samples: {len(features)}")
    print(f"   - Unique classes found: {np.unique(labels)}")
    print(f"   - Target classes: {num_classes} (shipsear 5分类)")
    
    # TSNE 核心配置（适配5分类数据分布）
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 10))
    
    # 5分类专属配色（更清晰区分5个类别）
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 5种鲜明颜色
    
    # 为每个类别单独画点（仅遍历5个类别）
    for i in range(num_classes):
        idx = labels == i
        if np.sum(idx) > 0:
            plt.scatter(
                features_2d[idx, 0], 
                features_2d[idx, 1], 
                c=[colors[i]], 
                label=f'Class {i}',
                s=20,  # 稍大的点更易区分
                alpha=0.85,
                edgecolors='k',
                linewidths=0.3
            )
    
    plt.title(title, fontsize=16)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved shipsear 5分类 t-SNE plot: {save_path}")
 
def plot_dynamic_trajectory(student_seq, teacher_seq, title, save_path):
    """绘制师生模型动态轨迹（适配shipsear数据特征）"""
    student_norms = np.linalg.norm(student_seq, axis=1)
    teacher_norms = np.linalg.norm(teacher_seq, axis=1)
    
    # 轨迹对齐（适配shipsear数据的时间步长）
    if len(teacher_norms) != len(student_norms):
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, len(teacher_norms))
        x_new = np.linspace(0, 1, len(student_norms))
        f = interp1d(x_old, teacher_norms, kind='linear')
        teacher_norms_aligned = f(x_new)
    else:
        teacher_norms_aligned = teacher_norms
 
    plt.figure(figsize=(10, 6))
    # 更醒目的配色和样式
    plt.plot(student_norms, label='Student Trajectory (LNN)', 
             color='#1f77b4', marker='o', markersize=5, linewidth=2.5)
    plt.plot(teacher_norms_aligned, label='Teacher Trajectory (CNN, Aligned)', 
             color='#d62728', linestyle='--', marker='x', markersize=5, linewidth=2.5)
    
    plt.title(title, fontsize=14)
    plt.xlabel("Time Step (Normalized)", fontsize=12)
    plt.ylabel("Feature L2 Norm", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved shipsear trajectory plot: {save_path}")
 
def main():
    parser = argparse.ArgumentParser(description='Feature Visualization for shipsear (5分类)')
    
    # === 核心修改：shipsear 专属路径配置 ===
    parser.add_argument('--config', type=str, 
                        default='/media/hdd1/fubohan/Code/UATR/checkpoints/ablation_shipsear/num_epochs200_weight_decay0.0001_batch_size20_lr0.0005_distill_typeMTSKD_Temp_freezeTrue_USE_DYNAMIC_DISTILL_WEIGHTFalse_p_encoder0.355_p_classifier0.255/config.yaml')
    parser.add_argument('--checkpoint', type=str, 
                        default='/media/hdd1/fubohan/Code/UATR/checkpoints/ablation_shipsear/num_epochs200_weight_decay0.0001_batch_size20_lr0.0005_distill_typeMTSKD_Temp_freezeTrue_USE_DYNAMIC_DISTILL_WEIGHTFalse_p_encoder0.355_p_classifier0.255/best_student.pth')
    parser.add_argument('--output_dir', type=str, default='results/experiment6_shipsear')  # 输出目录区分
    parser.add_argument('--max_samples', type=int, default=4000, help='最大采样数量（适配shipsear数据集规模）')
    
    args = parser.parse_args()
 
    setup_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
 
    # 1. 加载shipsear配置
    print(f"\n📂 Loading shipsear config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 强制指定5分类（防止配置文件读取异常）
    num_classes = 5
    print(f"🔍 Set shipsear num_classes = {num_classes}")
 
    # 2. 加载shipsear训练好的模型
    print("\n📦 Building model and loading shipsear weights...")
    model = AudioDistillationModel(
        num_classes=num_classes,
        teacher_pretrained=config['model']['teacher']['pretrained'],
        freeze_teacher=config['model']['teacher']['freeze'],
        teacher_checkpoint=config['model']['teacher']['checkpoint_path'],
        p_encoder=config['model']['student']['p_encoder'],
        p_classifier=config['model']['student']['p_classifier']
    ).to(device)
 
    # 加载权重（适配shipsear的checkpoint格式）
    ckpt = torch.load(args.checkpoint, map_location=device)
    if 'student_state_dict' in ckpt:
        model.student.load_state_dict(ckpt['student_state_dict'])
    else:
        model.student.load_state_dict(ckpt, strict=False)
    model.eval()
 
    # 3. 加载shipsear验证集数据
    print("\n📂 Loading shipsear validation data...")
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
 
    # 4. 提取shipsear特征
    print("\n🚀 Extracting shipsear features...")
    all_student_features = []
    all_labels = []
    
    saved_student_seq = None
    saved_teacher_seq = None
 
    with torch.no_grad():
        for batch_idx, (input_data, labels) in enumerate(val_loader):
            inputs = get_inputs(input_data, config['dataset']['data_type'], device)
            
            outputs = model(inputs)
            
            student_logits = outputs[0]
            x_encoder = outputs[6]
            output_cnn_features = outputs[8]
            
            all_student_features.append(student_logits.cpu().numpy())
            all_labels.append(labels.numpy())
            
            # 保存首条轨迹用于可视化
            if saved_student_seq is None:
                saved_student_seq = x_encoder[0].cpu().numpy()
                saved_teacher_seq = output_cnn_features[0].cpu().numpy()
 
            # 采样终止条件（适配5分类数据）
            current_count = (batch_idx + 1) * val_loader.batch_size
            unique_labels = np.unique(np.concatenate(all_labels))
            
            if current_count >= args.max_samples and len(unique_labels) == num_classes:
                print(f"   Collected {current_count} shipsear samples with all {num_classes} classes. Stopping extraction.")
                break
 
    # 5. 生成shipsear专属可视化结果
    all_student_features = np.concatenate(all_student_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # 绘制5分类t-SNE图
    plot_tsne(
        features=all_student_features,
        labels=all_labels,
        title="ShipSEAR: Student Feature Distribution (t-SNE, 5 Classes)",
        save_path=os.path.join(args.output_dir, "fig1_tsne_shipsear_5classes.png"),
        num_classes=num_classes
    )
 
    # 绘制动态轨迹图
    if saved_student_seq is not None and saved_teacher_seq is not None:
        plot_dynamic_trajectory(
            student_seq=saved_student_seq,
            teacher_seq=saved_teacher_seq,
            title="ShipSEAR: Dynamic Trajectory Alignment (5 Classes)",
            save_path=os.path.join(args.output_dir, "fig2_trajectory_shipsear.png")
        )
 
    print(f"\n✨ ShipSEAR (5分类) Visualization Complete!")
    print(f"📁 Results saved to: {args.output_dir}")
 
if __name__ == '__main__':
    main()