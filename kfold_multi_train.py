#!/usr/bin/env python
"""
多组K-Fold划分 - 完整训练和对比工具

一键生成多组划分，运行训练，对比结果，选择最优划分。

使用方法:
    # 生成3组K-Fold划分和完整训练流程
    python kfold_multi_train.py --all \\
        --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622 \\
        --num-splits 3 \\
        --gpus 4,5,6,7

    # 或只生成划分不训练
    python kfold_multi_train.py --generate \\
        --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622 \\
        --num-splits 3

    # 或只对比已有的训练结果
    python kfold_multi_train.py --compare \\
        --results-dir results/kfold_cv_shipsear_multi
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime
from pathlib import Path


class MultiKFoldTrainer:
    """多组K-Fold训练管理器"""

    def __init__(self, data_dir, output_base_dir="results", num_splits=3, gpus="4,5,6,7"):
        """
        参数:
            data_dir: 数据目录
            output_base_dir: 输出基础目录
            num_splits: 生成的划分组数
            gpus: GPU编号
        """
        self.data_dir = data_dir
        self.output_base_dir = output_base_dir
        self.splits_dir = os.path.join(output_base_dir, "kfold_splits_multi")
        self.results_dir = os.path.join(output_base_dir, "kfold_cv_shipsear_multi")
        self.num_splits = num_splits
        self.gpus = gpus
        self.split_configs = []

    def generate_multiple_splits(self):
        """生成多组K-Fold划分"""
        print("\n" + "="*80)
        print("📊 第1步: 生成多组K-Fold划分")
        print("="*80)

        cmd = (
            f"python kfold_multi_split_generator.py "
            f"--data-dir {self.data_dir} "
            f"--output-dir {self.splits_dir} "
            f"--num-splits {self.num_splits}"
        )

        print(f"\n运行命令: {cmd}\n")
        result = subprocess.run(cmd, shell=True)

        if result.returncode != 0:
            print("❌ 生成K-Fold划分失败")
            return False

        # 读取生成的split配置
        summary_file = os.path.join(self.splits_dir, "splits_summary.json")
        if os.path.exists(summary_file):
            with open(summary_file) as f:
                config = json.load(f)
                self.split_configs = list(config['splits'].values())
                print(f"\n✓ 已生成 {len(self.split_configs)} 组划分")
        else:
            print("⚠️  未找到splits_summary.json")

        return True

    def train_split(self, split_id, split_config):
        """训练单组划分"""
        split_name = split_config['split_name']
        split_dir = split_config['output_dir']

        print(f"\n{'='*80}")
        print(f"【第 {split_id+1}/{len(self.split_configs)} 组】训练: {split_name}")
        print(f"{'='*80}")

        # 该split对应的结果目录
        result_dir = os.path.join(
            self.results_dir,
            split_name
        )

        cmd = (
            f"python kfold_shipsear_integration.py "
            f"--train-all "
            f"--gpus {self.gpus} "
            f"--splits-dir {split_dir} "
            f"--checkpoints-dir {os.path.join(self.results_dir, split_name, 'checkpoints')} "
            f"--results-dir {result_dir}"
        )

        print(f"\n运行命令: {cmd}\n")
        result = subprocess.run(cmd, shell=True)

        if result.returncode != 0:
            print(f"✗ {split_name} 训练失败")
            return False
        else:
            print(f"✓ {split_name} 训练成功")
            return True

    def train_all_splits(self):
        """训练所有划分"""
        if not self.split_configs:
            print("❌ 未找到split配置")
            return False

        print("\n" + "="*80)
        print("🔥 第2步: 训练所有划分")
        print("="*80)

        success_count = 0
        for split_id, split_config in enumerate(self.split_configs):
            if self.train_split(split_id, split_config):
                success_count += 1

        print("\n" + "="*80)
        print(f"✨ 训练完成: {success_count}/{len(self.split_configs)} 组成功")
        print("="*80)

        return success_count > 0

    def compare_results(self):
        """对比所有划分的结果"""
        print("\n" + "="*80)
        print("📊 第3步: 对比分析结果")
        print("="*80)

        cmd = (
            f"python kfold_comparison_analyzer.py "
            f"--results-dir {self.results_dir}"
        )

        print(f"\n运行命令: {cmd}\n")
        result = subprocess.run(cmd, shell=True)

        return result.returncode == 0

    def print_summary(self):
        """打印总结"""
        print("\n" + "="*80)
        print("✨ 多组K-Fold训练流程完成!")
        print("="*80)

        print("\n📂 输出位置:")
        print(f"  K-Fold划分: {self.splits_dir}")
        print(f"  训练结果:  {self.results_dir}")

        print("\n📊 生成的结果:")
        print(f"  - 划分对比报告: {os.path.join(self.splits_dir, 'splits_comparison.txt')}")
        print(f"  - 性能对比报告: {os.path.join(self.results_dir, 'comparison_report.txt')}")

        print("\n📈 后续操作:")
        print("  1. 查看对比报告，找出最优划分")
        print("  2. 使用最优划分进行最终训练和验证")
        print("  3. 在论文中记录使用的划分配置和性能")

        print("\n💾 保存到版本控制:")
        print(f"  git add {self.splits_dir}/")
        print(f"  git add {self.results_dir}/")
        print("  git commit -m 'Add multi-split K-Fold results'")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="多组K-Fold划分完整训练流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 完整流程: 生成划分 -> 训练 -> 对比
  python kfold_multi_train.py --all \\
      --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622 \\
      --num-splits 3 \\
      --gpus 4,5,6,7

  # 只生成划分
  python kfold_multi_train.py --generate \\
      --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622 \\
      --num-splits 3

  # 只训练（需要先生成划分）
  python kfold_multi_train.py --train

  # 只对比
  python kfold_multi_train.py --compare \\
      --results-dir results/kfold_cv_shipsear_multi

  # 查看划分对比
  cat results/kfold_splits_multi/splits_comparison.txt

  # 查看性能对比
  cat results/kfold_cv_shipsear_multi/comparison_report.txt
        """
    )

    parser.add_argument("--all", action="store_true",
                        help="完整流程: 生成 -> 训练 -> 对比")
    parser.add_argument("--generate", action="store_true",
                        help="只生成划分")
    parser.add_argument("--train", action="store_true",
                        help="只训练")
    parser.add_argument("--compare", action="store_true",
                        help="只对比结果")

    parser.add_argument("--data-dir", type=str, default="./data",
                        help="数据目录")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="输出基础目录")
    parser.add_argument("--num-splits", type=int, default=3,
                        help="划分组数")
    parser.add_argument("--gpus", type=str, default="4,5,6,7",
                        help="GPU编号")
    parser.add_argument("--results-dir", type=str,
                        help="结果目录（仅用于--compare）")

    args = parser.parse_args()

    # 检查参数
    if not args.all and not args.generate and not args.train and not args.compare:
        parser.print_help()
        return

    # 创建训练器
    trainer = MultiKFoldTrainer(
        data_dir=args.data_dir,
        output_base_dir=args.output_dir,
        num_splits=args.num_splits,
        gpus=args.gpus
    )

    # 执行操作
    if args.all:
        print("\n🚀 开始多组K-Fold完整流程...")
        if not trainer.generate_multiple_splits():
            return
        if not trainer.train_all_splits():
            return
        if not trainer.compare_results():
            return
        trainer.print_summary()

    elif args.generate:
        print("\n📊 生成多组K-Fold划分...")
        if trainer.generate_multiple_splits():
            print("\n✓ 划分生成完成")

    elif args.train:
        print("\n🔥 训练所有划分...")
        # 读取已生成的splits
        summary_file = os.path.join(trainer.splits_dir, "splits_summary.json")
        if os.path.exists(summary_file):
            with open(summary_file) as f:
                config = json.load(f)
                trainer.split_configs = list(config['splits'].values())
        if trainer.train_all_splits():
            print("\n✓ 训练完成")

    elif args.compare:
        print("\n📊 对比分析结果...")
        if args.results_dir:
            # 自定义结果目录
            cmd = f"python kfold_comparison_analyzer.py --results-dir {args.results_dir}"
        else:
            # 使用默认结果目录
            cmd = f"python kfold_comparison_analyzer.py --results-dir {trainer.results_dir}"

        result = subprocess.run(cmd, shell=True)
        if result.returncode == 0:
            print("\n✓ 对比完成")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
    else:
        main()
