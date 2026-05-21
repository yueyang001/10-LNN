"""
通用t-SNE特征可视化脚本
支持多种特征来源和可视化模式
"""
import argparse
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, RandomSampler
import seaborn as sns

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


def extract_features(model, data_loader, device, config, max_samples=None, feature_layer='logits'):
    """
    提取模型特征
    feature_layer options: 'logits', 'encoder', 'cnn_features', 'combined'
    """
    all_features = []
    all_labels = []
    additional_outputs = {}

    model.eval()
    with torch.no_grad():
        for batch_idx, (input_data, labels) in enumerate(data_loader):
            inputs = get_inputs(input_data, config['dataset']['data_type'], device)
            outputs = model(inputs)

            # 提取不同层的特征
            if feature_layer == 'logits':
                features = outputs[0]  # student_logits
            elif feature_layer == 'encoder':
                features = outputs[6]  # x_encoder (from student)
            elif feature_layer == 'cnn_features':
                features = outputs[8]  # output_cnn_features (from teacher)
            elif feature_layer == 'combined':
                # 组合多层特征
                student_features = outputs[6]  # encoder
                teacher_features = outputs[8]  # cnn features
                # 简单拼接
                features = torch.cat([student_features, teacher_features], dim=1)
            else:
                raise ValueError(f"Unknown feature_layer: {feature_layer}")

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

            if batch_idx == 0:
                additional_outputs['first_encoder'] = outputs[6][0].cpu().numpy()
                additional_outputs['first_cnn'] = outputs[8][0].cpu().numpy()

            current_count = (batch_idx + 1) * data_loader.batch_size
            if max_samples and current_count >= max_samples:
                print(f"Reached max_samples: {current_count}")
                break

    features_array = np.concatenate(all_features, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)

    print(f"Features shape: {features_array.shape}")
    print(f"Labels shape: {labels_array.shape}")

    return features_array, labels_array, additional_outputs


def apply_tsne(features, perplexity=30, random_state=42, learning_rate='auto'):
    """应用t-SNE降维"""
    print(f"Applying t-SNE with {features.shape[0]} samples, {features.shape[1]} dimensions...")

    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 调整perplexity
    perplexity = min(perplexity, (features.shape[0] - 1) // 3)
    print(f"Adjusted perplexity: {perplexity}")

    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=perplexity,
        n_iter=1000,
        learning_rate=learning_rate,
        init='pca',
        verbose=1
    )

    features_2d = tsne.fit_transform(features_scaled)
    return features_2d


def plot_tsne_by_class(features_2d, labels, num_classes, title, save_path):
    """按类别绘制t-SNE图"""
    plt.figure(figsize=(12, 10))

    # 使用高对比度颜色
    if num_classes <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    elif num_classes <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
    else:
        # 对于更多类别，使用HSV色图
        colors = plt.cm.hsv(np.linspace(0, 1, num_classes))

    # 为每个类别绘制点
    for i in range(num_classes):
        idx = labels == i
        if np.sum(idx) > 0:
            plt.scatter(
                features_2d[idx, 0],
                features_2d[idx, 1],
                c=[colors[i]],
                label=f'Class {i}',
                s=20,
                alpha=0.7,
                edgecolors='black',
                linewidths=0.3
            )

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(fontsize=10, loc='best', ncol=2)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved class-colored plot: {save_path}")


def plot_tsne_continuous(features_2d, values, title, save_path, cmap='viridis'):
    """使用连续值绘制t-SNE图（如置信度）"""
    plt.figure(figsize=(12, 10))

    scatter = plt.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=values,
        cmap=cmap,
        s=20,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.3
    )

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.colorbar(scatter, label='Value')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved continuous-colored plot: {save_path}")


def plot_tsne_density(features_2d, labels, num_classes, title, save_path):
    """绘制带密度的t-SNE图"""
    fig, ax = plt.subplots(figsize=(12, 10))

    if num_classes <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, num_classes))

    # 绘制KDE密度背景
    from scipy.stats import gaussian_kde

    try:
        # 计算整体密度
        xy = np.vstack([features_2d[:, 0], features_2d[:, 1]])
        z = gaussian_kde(xy)(xy)

        # 按密度排序（低密度点先绘制）
        idx = z.argsort()
        x_sorted = features_2d[idx, 0]
        y_sorted = features_2d[idx, 1]
        z_sorted = z[idx]
        labels_sorted = labels[idx]

        # 绘制所有点
        for i in range(num_classes):
            idx = labels_sorted == i
            if np.sum(idx) > 0:
                ax.scatter(
                    x_sorted[idx],
                    y_sorted[idx],
                    c=[colors[i]],
                    label=f'Class {i}',
                    s=20,
                    alpha=0.6,
                    edgecolors='black',
                    linewidths=0.3
                )
    except Exception as e:
        print(f"Warning: Could not compute density. {e}")
        # 回退到简单绘图
        plot_tsne_by_class(features_2d, labels, num_classes, title, save_path)
        return

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.legend(fontsize=10, loc='best', ncol=2)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved density plot: {save_path}")


def create_comparison_plot(features_2d_list, labels, num_classes, titles, save_path):
    """创建多个t-SNE图的对比"""
    n_plots = len(features_2d_list)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    if num_classes <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, num_classes))

    for plot_idx, (features_2d, title) in enumerate(zip(features_2d_list, titles)):
        ax = axes[plot_idx]

        for i in range(num_classes):
            idx = labels == i
            if np.sum(idx) > 0:
                ax.scatter(
                    features_2d[idx, 0],
                    features_2d[idx, 1],
                    c=[colors[i]],
                    label=f'Class {i}' if plot_idx == 0 else '',
                    s=15,
                    alpha=0.7,
                    edgecolors='black',
                    linewidths=0.2
                )

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Dim 1", fontsize=10)
        ax.set_ylabel("Dim 2", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)

    # 只在第一个图上显示图例
    if n_plots > 0:
        axes[0].legend(fontsize=8, loc='best', ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved comparison plot: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='通用t-SNE特征可视化')

    parser.add_argument('--config', type=str, help='模型配置文件路径')
    parser.add_argument('--checkpoint', type=str, help='模型检查点路径')
    parser.add_argument('--output_dir', type=str, default='results/tsne_visualization',
                        help='输出目录')
    parser.add_argument('--max_samples', type=int, default=2000,
                        help='最大采样数量')
    parser.add_argument('--feature_layer', type=str, default='logits',
                        choices=['logits', 'encoder', 'cnn_features', 'combined'],
                        help='特征层选择')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='t-SNE perplexity参数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批大小')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载配置
    if not args.config:
        raise ValueError("必须提供 --config 参数")

    print(f"📂 Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    num_classes = config['model']['num_classes']
    print(f"Number of classes: {num_classes}")

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

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        if 'student_state_dict' in ckpt:
            model.student.load_state_dict(ckpt['student_state_dict'])
        else:
            model.student.load_state_dict(ckpt, strict=False)
        print(f"✅ Loaded checkpoint: {args.checkpoint}")

    model.eval()

    # 3. 加载数据
    print("📂 Loading validation data...")
    val_dataset = AudioDataset(
        data_dir=config['dataset']['data_dir'],
        data_flag='validation',
        data_type=config['dataset']['data_type'],
        transform=validation_test_transform
    )

    sampler = RandomSampler(val_dataset, replacement=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4
    )

    # 4. 提取特征
    print(f"🚀 Extracting {args.feature_layer} features...")
    features, labels, additional = extract_features(
        model, val_loader, device, config,
        max_samples=args.max_samples,
        feature_layer=args.feature_layer
    )

    # 5. 应用t-SNE
    print("\n📉 Computing t-SNE...")
    features_2d = apply_tsne(features, perplexity=args.perplexity)

    # 6. 生成可视化
    print("\n🎨 Generating visualizations...")

    # 按类别着色
    plot_tsne_by_class(
        features_2d, labels, num_classes,
        f"t-SNE: {args.feature_layer.capitalize()} Features (Colored by Class)",
        os.path.join(args.output_dir, f"tsne_{args.feature_layer}_by_class.png")
    )

    # 按样本索引着色
    plot_tsne_continuous(
        features_2d, np.arange(len(labels)),
        f"t-SNE: {args.feature_layer.capitalize()} Features (Colored by Sample Index)",
        os.path.join(args.output_dir, f"tsne_{args.feature_layer}_by_index.png"),
        cmap='rainbow'
    )

    print(f"\n✨ Visualization complete! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
