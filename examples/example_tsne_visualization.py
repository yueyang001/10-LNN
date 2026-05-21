#!/usr/bin/env python
"""
t-SNE特征可视化 - 完整示例
演示如何使用t-SNE可视化不同的特征
"""
import os
import sys
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, RandomSampler

# 导入项目模块
try:
    from models.distillation import AudioDistillationModel
    from datasets.audio_dataset import AudioDataset, validation_test_transform
except ImportError:
    print("错误: 无法导入项目模块")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)


class FeatureExtractor:
    """特征提取器"""

    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config

    def get_inputs(self, input_data):
        """获取输入数据"""
        data_type = self.config['dataset']['data_type']
        if '@' not in data_type:
            return input_data[0].to(self.device)
        else:
            return [data.to(self.device) for data in input_data]

    def extract_features(self, data_loader, feature_layer='logits', max_samples=None):
        """
        提取特征
        feature_layer: 'logits', 'encoder', 'cnn_features', 'combined'
        """
        all_features = []
        all_labels = []

        self.model.eval()

        with torch.no_grad():
            for batch_idx, (input_data, labels) in enumerate(data_loader):
                inputs = self.get_inputs(input_data)
                outputs = self.model(inputs)

                # 提取特征
                if feature_layer == 'logits':
                    features = outputs[0]
                elif feature_layer == 'encoder':
                    features = outputs[6]
                elif feature_layer == 'cnn_features':
                    features = outputs[8]
                elif feature_layer == 'combined':
                    features = torch.cat([outputs[6], outputs[8]], dim=1)
                else:
                    raise ValueError(f"Unknown feature layer: {feature_layer}")

                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())

                current_count = (batch_idx + 1) * data_loader.batch_size
                if max_samples and current_count >= max_samples:
                    break

        features_array = np.concatenate(all_features, axis=0)
        labels_array = np.concatenate(all_labels, axis=0)

        return features_array, labels_array


class TSNEVisualizer:
    """t-SNE可视化器"""

    @staticmethod
    def compute_tsne(features, perplexity=30, random_state=42):
        """计算t-SNE"""
        print(f"应用t-SNE (n_samples={features.shape[0]}, n_features={features.shape[1]})...")

        # 标准化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # 调整perplexity
        perplexity = min(perplexity, (features.shape[0] - 1) // 3)

        tsne = TSNE(
            n_components=2,
            random_state=random_state,
            perplexity=perplexity,
            n_iter=1000,
            learning_rate='auto',
            init='pca',
            verbose=1
        )

        features_2d = tsne.fit_transform(features_scaled)
        return features_2d

    @staticmethod
    def plot_by_class(features_2d, labels, num_classes, title, save_path):
        """按类别绘制"""
        plt.figure(figsize=(12, 10))

        if num_classes <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
        elif num_classes <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
        else:
            colors = plt.cm.hsv(np.linspace(0, 1, num_classes))

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
        plt.legend(fontsize=9, loc='best', ncol=2)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ 已保存: {save_path}")

    @staticmethod
    def plot_by_continuous(features_2d, values, title, save_path, cmap='viridis'):
        """使用连续值绘制"""
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

        print(f"✓ 已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='t-SNE特征可视化 - 完整示例',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 可视化logits特征
  python examples/example_tsne_visualization.py \\
    --config configs/train_distillation_deepship.yaml \\
    --feature logits

  # 可视化编码器特征
  python examples/example_tsne_visualization.py \\
    --config configs/train_distillation_deepship.yaml \\
    --feature encoder

  # 自定义参数
  python examples/example_tsne_visualization.py \\
    --config configs/train_distillation_deepship.yaml \\
    --feature combined \\
    --max_samples 3000 \\
    --perplexity 50
        """
    )

    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型检查点路径 (可选)')
    parser.add_argument('--feature', type=str, default='logits',
                        choices=['logits', 'encoder', 'cnn_features', 'combined'],
                        help='特征层 (默认: logits)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录 (默认: results/tsne_examples)')
    parser.add_argument('--max_samples', type=int, default=2000,
                        help='最大样本数 (默认: 2000)')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='tSNE perplexity (默认: 30)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批大小 (默认: 32)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    output_dir = args.output or 'results/tsne_examples'
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载配置
    print(f"📂 加载配置: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    num_classes = config['model']['num_classes']
    print(f"类别数: {num_classes}\n")

    # 2. 构建模型
    print("📦 构建模型...")
    model = AudioDistillationModel(
        num_classes=num_classes,
        teacher_pretrained=config['model']['teacher']['pretrained'],
        freeze_teacher=config['model']['teacher']['freeze'],
        teacher_checkpoint=config['model']['teacher']['checkpoint_path'],
        p_encoder=config['model']['student']['p_encoder'],
        p_classifier=config['model']['student']['p_classifier']
    ).to(device)

    if args.checkpoint:
        print(f"加载检查点: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        if 'student_state_dict' in ckpt:
            model.student.load_state_dict(ckpt['student_state_dict'])
        else:
            model.student.load_state_dict(ckpt, strict=False)

    model.eval()
    print("✓ 模型准备完毕\n")

    # 3. 加载数据
    print("📂 加载验证数据...")
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
    print(f"✓ 数据集大小: {len(val_dataset)}\n")

    # 4. 提取特征
    print(f"🚀 提取 {args.feature} 特征...")
    extractor = FeatureExtractor(model, device, config)
    features, labels = extractor.extract_features(
        val_loader,
        feature_layer=args.feature,
        max_samples=args.max_samples
    )
    print(f"✓ 特征形状: {features.shape}\n")

    # 5. 计算t-SNE
    print("📉 计算t-SNE...")
    visualizer = TSNEVisualizer()
    features_2d = visualizer.compute_tsne(
        features,
        perplexity=args.perplexity,
        random_state=args.seed
    )
    print("✓ t-SNE完成\n")

    # 6. 生成可视化
    print("🎨 生成可视化...")

    # 按类别着色
    visualizer.plot_by_class(
        features_2d, labels, num_classes,
        f"t-SNE: {args.feature.upper()} Features (按类别着色)",
        os.path.join(output_dir, f"tsne_{args.feature}_by_class.png")
    )

    # 按样本索引着色
    visualizer.plot_by_continuous(
        features_2d,
        np.arange(len(labels)),
        f"t-SNE: {args.feature.upper()} Features (按样本索引着色)",
        os.path.join(output_dir, f"tsne_{args.feature}_by_index.png"),
        cmap='rainbow'
    )

    print(f"\n✨ 完成! 结果保存到: {output_dir}")


if __name__ == '__main__':
    main()
