"""
十折叠交叉验证脚本
- 扫描数据目录结构
- 生成平衡的十折叠数据集划分
- 保存详细的划分结果到txt文档
- 支持复现性（固定随机种子）
"""

import os
import glob
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict


class KFoldCrossValidator:
    def __init__(self, data_dir, output_dir="results/kfold_splits", seed=42):
        """
        参数:
            data_dir: 数据根目录路径
            output_dir: 输出划分结果的目录
            seed: 随机种子，用于复现
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.seed = seed
        self.n_splits = 10

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 初始化数据结构
        self.all_samples = []  # 所有样本列表
        self.category_map = {}  # 类别名称到编号的映射
        self.fold_splits = {}  # 折叠划分结果

    def scan_dataset(self, data_flag="train"):
        """
        扫描指定数据集目录（train/validation/test）

        参数:
            data_flag: 数据标签 ("train", "validation", "test")

        返回:
            样本列表: [(wav_path, category_idx, category_name), ...]
        """
        dataset_dir = os.path.join(self.data_dir, data_flag)
        wav_dir = os.path.join(dataset_dir, "wav")

        if not os.path.exists(wav_dir):
            print(f"❌ 错误: 未找到 {wav_dir}")
            return []

        samples = []
        category_names = sorted(os.listdir(wav_dir))

        # 建立类别映射
        for idx, cat_name in enumerate(category_names):
            self.category_map[cat_name] = idx

        # 遍历每个类别
        for category_name in category_names:
            category_wav_dir = os.path.join(wav_dir, category_name)
            wav_files = sorted(glob.glob(os.path.join(category_wav_dir, "*.wav")))

            for wav_path in wav_files:
                category_idx = self.category_map[category_name]
                samples.append((wav_path, category_idx, category_name))

        print(f"✓ 扫描完成: {data_flag} - 共 {len(samples)} 个样本")
        print(f"  类别数: {len(category_names)}")
        print(f"  类别列表: {category_names}")

        return samples

    def generate_folds(self, samples):
        """
        使用 StratifiedKFold 生成平衡的十折叠划分

        参数:
            samples: 样本列表 [(wav_path, category_idx, category_name), ...]

        返回:
            fold_splits: {fold_idx: {"train": [...], "val": [...]}, ...}
        """
        if not samples:
            print("❌ 错误: 没有样本数据")
            return {}

        # 提取样本路径和标签
        sample_paths = [s[0] for s in samples]
        labels = [s[1] for s in samples]

        # 创建StratifiedKFold分割器
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        fold_splits = {}
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(sample_paths, labels)):
            train_samples = [samples[i] for i in train_idx]
            val_samples = [samples[i] for i in val_idx]

            fold_splits[fold_idx] = {
                "train": train_samples,
                "val": val_samples
            }

            # 统计每折的类别分布
            train_labels = [s[1] for s in train_samples]
            val_labels = [s[1] for s in val_samples]

            print(f"  Fold {fold_idx}: train={len(train_samples)}, val={len(val_samples)}")
            print(f"    train标签分布: {np.bincount(train_labels).tolist()}")
            print(f"    val标签分布: {np.bincount(val_labels).tolist()}")

        self.fold_splits = fold_splits
        return fold_splits

    def save_fold_split(self, fold_idx, output_file):
        """
        将单个折叠的划分结果保存到txt文件

        参数:
            fold_idx: 折叠索引
            output_file: 输出文件路径
        """
        if fold_idx not in self.fold_splits:
            print(f"❌ 错误: 折叠 {fold_idx} 不存在")
            return

        fold_data = self.fold_splits[fold_idx]

        with open(output_file, "w", encoding="utf-8") as f:
            # 头部信息
            f.write("=" * 80 + "\n")
            f.write(f"十折叠交叉验证 - Fold {fold_idx}\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"随机种子: {self.seed}\n")
            f.write("=" * 80 + "\n\n")

            # 统计信息
            train_samples = fold_data["train"]
            val_samples = fold_data["val"]
            train_labels = [s[1] for s in train_samples]
            val_labels = [s[1] for s in val_samples]

            f.write("【数据集统计】\n")
            f.write(f"训练集样本数: {len(train_samples)}\n")
            f.write(f"验证集样本数: {len(val_samples)}\n")
            f.write(f"训练集标签分布: {np.bincount(train_labels).tolist()}\n")
            f.write(f"验证集标签分布: {np.bincount(val_labels).tolist()}\n")
            f.write("\n")

            # 训练集详情
            f.write("-" * 80 + "\n")
            f.write("【训练集 (Train Set)】\n")
            f.write("-" * 80 + "\n")
            for idx, (wav_path, cat_idx, cat_name) in enumerate(train_samples, 1):
                f.write(f"{idx:6d} | {cat_idx} | {cat_name:20s} | {wav_path}\n")

            f.write("\n")

            # 验证集详情
            f.write("-" * 80 + "\n")
            f.write("【验证集 (Validation Set)】\n")
            f.write("-" * 80 + "\n")
            for idx, (wav_path, cat_idx, cat_name) in enumerate(val_samples, 1):
                f.write(f"{idx:6d} | {cat_idx} | {cat_name:20s} | {wav_path}\n")

            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("复现说明:\n")
            f.write(f"使用此文件中的样本路径列表，结合随机种子 {self.seed}\n")
            f.write("可以完全复现该Fold的数据划分结果。\n")
            f.write("=" * 80 + "\n")

    def save_all_splits(self, suffix=""):
        """
        保存所有折叠的划分结果

        参数:
            suffix: 文件名后缀
        """
        if not self.fold_splits:
            print("❌ 错误: 没有折叠数据可保存")
            return

        # 生成总体汇总文件
        summary_file = os.path.join(self.output_dir, f"kfold_summary{suffix}.txt")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("十折叠交叉验证 - 总体汇总\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"随机种子: {self.seed}\n")
            f.write(f"数据目录: {self.data_dir}\n")
            f.write("=" * 80 + "\n\n")

            # 汇总每个折叠的统计
            for fold_idx in sorted(self.fold_splits.keys()):
                fold_data = self.fold_splits[fold_idx]
                train_samples = fold_data["train"]
                val_samples = fold_data["val"]
                train_labels = [s[1] for s in train_samples]
                val_labels = [s[1] for s in val_samples]

                f.write(f"Fold {fold_idx}:\n")
                f.write(f"  训练集: {len(train_samples)} 样本 | 标签分布: {np.bincount(train_labels).tolist()}\n")
                f.write(f"  验证集: {len(val_samples)} 样本 | 标签分布: {np.bincount(val_labels).tolist()}\n")
                f.write(f"  详细文件: kfold_fold{fold_idx:02d}{suffix}.txt\n\n")

        # 保存每个折叠的详细文件
        for fold_idx in sorted(self.fold_splits.keys()):
            fold_file = os.path.join(self.output_dir, f"kfold_fold{fold_idx:02d}{suffix}.txt")
            self.save_fold_split(fold_idx, fold_file)
            print(f"✓ 已保存 Fold {fold_idx}: {fold_file}")

        print(f"\n✓ 已保存总体汇总: {summary_file}")
        return summary_file

    def save_split_indices(self, suffix=""):
        """
        保存索引形式的划分结果（便于Python脚本加载）
        输出格式: fold_idx | sample_idx | split_type(train/val)
        """
        indices_file = os.path.join(self.output_dir, f"kfold_indices{suffix}.txt")

        with open(indices_file, "w", encoding="utf-8") as f:
            f.write("fold_idx | sample_idx | split_type\n")
            f.write("-" * 40 + "\n")

            sample_idx = 0
            for fold_idx in sorted(self.fold_splits.keys()):
                fold_data = self.fold_splits[fold_idx]

                # 记录训练集索引
                for _ in fold_data["train"]:
                    f.write(f"{fold_idx} | {sample_idx} | train\n")
                    sample_idx += 1

                # 记录验证集索引
                for _ in fold_data["val"]:
                    f.write(f"{fold_idx} | {sample_idx} | val\n")
                    sample_idx += 1

        print(f"✓ 已保存索引文件: {indices_file}")
        return indices_file


def main():
    """主函数：演示如何使用"""

    # ============ 配置 ============
    data_dir = '/media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622'  # 修改为你的数据目录

    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"❌ 错误: 数据目录 {data_dir} 不存在")
        print("请修改 data_dir 变量指向正确的数据路径")
        return

    # ============ 创建验证器 ============
    validator = KFoldCrossValidator(
        data_dir=data_dir,
        output_dir="results/kfold_splits",
        seed=42  # 固定随机种子保证复现性
    )

    # ============ 扫描数据集 ============
    print("\n📊 开始扫描数据集...")
    samples = validator.scan_dataset(data_flag="train")

    if not samples:
        print("❌ 无法获取样本数据")
        return

    # ============ 生成折叠划分 ============
    print("\n🔄 生成十折叠划分...")
    validator.generate_folds(samples)

    # ============ 保存结果 ============
    print("\n💾 保存划分结果...")
    validator.save_all_splits(suffix="")
    validator.save_split_indices(suffix="")

    print("\n✨ 完成! 所有文件已保存到:", validator.output_dir)
    print("\n📌 下次复现时使用相同的:")
    print(f"   - 数据目录: {data_dir}")
    print(f"   - 随机种子: {validator.seed}")


if __name__ == "__main__":
    main()
