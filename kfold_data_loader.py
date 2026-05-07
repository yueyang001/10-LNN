"""
K-Fold 50折叠数据加载器
用于在训练脚本中方便地加载划分好的K-Fold数据

使用示例:
    loader = KFoldDataLoader("results/kfold_splits/", fold_idx=0)
    train_samples = loader.get_train_samples()
    val_samples = loader.get_val_samples()
"""

import os
import re
from pathlib import Path


class KFoldDataLoader:
    """从保存的K-Fold文件中加载数据"""

    def __init__(self, split_dir, fold_idx):
        """
        参数:
            split_dir: 包含kfold_fold*.txt文件的目录
            fold_idx: 折叠索引 (0-9)
        """
        self.split_dir = split_dir
        self.fold_idx = fold_idx
        self.fold_file = os.path.join(split_dir, f"kfold_fold{fold_idx:02d}.txt")

        if not os.path.exists(self.fold_file):
            raise FileNotFoundError(f"找不到文件: {self.fold_file}")

        self.train_samples = []
        self.val_samples = []
        self._parse_fold_file()

    def _parse_fold_file(self):
        """解析K-Fold文件，提取训练集和验证集"""
        current_section = None

        with open(self.fold_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # 识别章节标题
                if "【训练集" in line:
                    current_section = "train"
                    continue
                elif "【验证集" in line:
                    current_section = "val"
                    continue
                elif line.startswith("---") or line.startswith("==="):
                    continue

                # 跳过空行和标题行
                if not line or "|" not in line:
                    continue

                # 解析数据行: idx | cat_idx | cat_name | path
                try:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 4:
                        try:
                            idx = int(parts[0])
                            cat_idx = int(parts[1])
                            cat_name = parts[2]
                            wav_path = parts[3]

                            sample = (wav_path, cat_idx, cat_name)

                            if current_section == "train":
                                self.train_samples.append(sample)
                            elif current_section == "val":
                                self.val_samples.append(sample)
                        except ValueError:
                            continue
                except:
                    continue

    def get_train_samples(self):
        """获取训练集样本列表"""
        return self.train_samples

    def get_val_samples(self):
        """获取验证集样本列表"""
        return self.val_samples

    def get_summary(self):
        """获取该折的统计摘要"""
        return {
            "fold_idx": self.fold_idx,
            "n_train": len(self.train_samples),
            "n_val": len(self.val_samples),
            "train_samples": self.train_samples,
            "val_samples": self.val_samples
        }


def load_kfold_splits(split_dir):
    """
    加载所有K-Fold划分

    返回:
        dict: {fold_idx: {"train": [...], "val": [...]}, ...}
    """
    splits = {}

    for fold_idx in range(50):
        try:
            loader = KFoldDataLoader(split_dir, fold_idx)
            splits[fold_idx] = {
                "train": loader.get_train_samples(),
                "val": loader.get_val_samples()
            }
        except FileNotFoundError:
            continue

    return splits


if __name__ == "__main__":
    # 示例使用
    split_dir = "results/kfold_splits/"

    # 加载第0折
    loader = KFoldDataLoader(split_dir, fold_idx=0)
    summary = loader.get_summary()

    print(f"Fold {summary['fold_idx']}:")
    print(f"  训练集: {summary['n_train']} 样本")
    print(f"  验证集: {summary['n_val']} 样本")
    print(f"\n训练集前3个样本:")
    for sample in summary['train_samples'][:3]:
        print(f"  {sample}")
