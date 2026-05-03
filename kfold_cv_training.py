"""
K-Fold交叉验证 + 模型训练 完整集成脚本

此脚本演示如何:
1. 加载K-Fold划分结果
2. 为每个Fold运行一次完整的训练
3. 收集所有Fold的结果进行统计分析

使用方式:
    python kfold_cv_training.py --fold 0 --config configs/train_config.yaml
    或
    python kfold_cv_training.py --all  # 运行所有10个Fold
"""

import os
import sys
import argparse
import csv
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from kfold_data_loader import KFoldDataLoader, load_kfold_splits


class KFoldTrainer:
    """K-Fold交叉验证训练器"""

    def __init__(self, splits_dir="results/kfold_splits", results_dir="results/kfold_cv_results"):
        """
        参数:
            splits_dir: K-Fold划分文件目录
            results_dir: 训练结果保存目录
        """
        self.splits_dir = splits_dir
        self.results_dir = results_dir
        self.results_csv = os.path.join(results_dir, "kfold_results.csv")

        os.makedirs(results_dir, exist_ok=True)

        # 初始化结果CSV
        if not os.path.exists(self.results_csv):
            with open(self.results_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "fold_idx",
                    "n_train",
                    "n_val",
                    "best_train_acc",
                    "best_val_acc",
                    "best_epoch",
                    "training_time",
                    "status",
                    "remarks"
                ])

    def get_fold_samples(self, fold_idx):
        """获取指定Fold的样本"""
        try:
            loader = KFoldDataLoader(self.splits_dir, fold_idx)
            return loader.get_train_samples(), loader.get_val_samples()
        except FileNotFoundError as e:
            print(f"❌ 错误: {e}")
            return None, None

    def train_fold(self, fold_idx, config, gpus="0"):
        """
        训练单个Fold

        参数:
            fold_idx: 折叠索引
            config: 训练配置字典
            gpus: GPU编号列表

        返回:
            训练结果字典
        """
        print(f"\n{'='*80}")
        print(f"K-Fold Training - Fold {fold_idx}/9")
        print(f"{'='*80}")

        # 获取该Fold的数据
        train_samples, val_samples = self.get_fold_samples(fold_idx)
        if train_samples is None:
            print(f"❌ 无法加载Fold {fold_idx}的数据")
            return None

        print(f"✓ 训练集: {len(train_samples)} 样本")
        print(f"✓ 验证集: {len(val_samples)} 样本")

        # 更新配置
        config["dataset"]["fold"] = fold_idx
        config["dataset"]["train_samples"] = train_samples
        config["dataset"]["val_samples"] = val_samples

        # 创建该Fold的输出目录
        fold_save_dir = os.path.join(self.results_dir, f"fold_{fold_idx:02d}")
        os.makedirs(fold_save_dir, exist_ok=True)
        config["save"]["save_dir"] = fold_save_dir

        # 这里应该调用你的训练函数
        # result = your_train_function(config, gpus)
        # 为了演示，这里使用模拟结果
        result = self._mock_train(fold_idx, fold_save_dir)

        # 保存结果到CSV
        self._save_result_to_csv(fold_idx, len(train_samples), len(val_samples), result)

        return result

    def _mock_train(self, fold_idx, save_dir):
        """模拟训练（用于演示）"""
        import random
        time_elapsed = random.uniform(30, 60)
        best_train_acc = random.uniform(0.85, 0.95)
        best_val_acc = random.uniform(0.80, 0.90)
        best_epoch = random.randint(50, 150)

        # 保存日志
        log_file = os.path.join(save_dir, "training.log")
        with open(log_file, "w") as f:
            f.write(f"Mock training for Fold {fold_idx}\n")
            f.write(f"Best train accuracy: {best_train_acc:.4f}\n")
            f.write(f"Best val accuracy: {best_val_acc:.4f}\n")
            f.write(f"Best epoch: {best_epoch}\n")

        return {
            "best_train_acc": best_train_acc,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "training_time": time_elapsed,
            "status": "success"
        }

    def _save_result_to_csv(self, fold_idx, n_train, n_val, result):
        """保存训练结果到CSV"""
        if result is None:
            with open(self.results_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([fold_idx, n_train, n_val, None, None, None, None, "failed", "无数据"])
        else:
            with open(self.results_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    fold_idx,
                    n_train,
                    n_val,
                    result.get("best_train_acc"),
                    result.get("best_val_acc"),
                    result.get("best_epoch"),
                    result.get("training_time"),
                    result.get("status"),
                    result.get("remarks", "")
                ])

    def train_all_folds(self, config, gpus="0"):
        """训练所有10个Fold"""
        results = {}
        for fold_idx in range(10):
            result = self.train_fold(fold_idx, config, gpus)
            results[fold_idx] = result

        # 生成总结报告
        self.generate_summary_report(results)

        return results

    def generate_summary_report(self, results):
        """生成汇总报告"""
        report_file = os.path.join(self.results_dir, "kfold_summary_report.txt")

        val_accs = [r["best_val_acc"] for r in results.values() if r is not None]

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("K-Fold交叉验证 - 训练总结报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            f.write("【验证集精度统计】\n")
            if val_accs:
                f.write(f"平均精度:  {np.mean(val_accs):.4f}\n")
                f.write(f"最高精度:  {np.max(val_accs):.4f}\n")
                f.write(f"最低精度:  {np.min(val_accs):.4f}\n")
                f.write(f"标准差:    {np.std(val_accs):.4f}\n\n")

            f.write("【各Fold精度】\n")
            for fold_idx in sorted(results.keys()):
                result = results[fold_idx]
                if result is not None:
                    f.write(f"Fold {fold_idx}: ")
                    f.write(f"train_acc={result['best_train_acc']:.4f}, ")
                    f.write(f"val_acc={result['best_val_acc']:.4f}, ")
                    f.write(f"epoch={result['best_epoch']}\n")
                else:
                    f.write(f"Fold {fold_idx}: FAILED\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("详细结果CSV: kfold_results.csv\n")
            f.write("=" * 80 + "\n")

        print(f"\n✓ 汇总报告已保存: {report_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="K-Fold交叉验证训练")
    parser.add_argument("--fold", type=int, default=None, help="训练指定的Fold (0-9)")
    parser.add_argument("--all", action="store_true", help="训练所有Fold")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml",
                        help="训练配置文件路径")
    parser.add_argument("--gpus", type=str, default="0", help="GPU编号")
    parser.add_argument("--splits-dir", type=str, default="results/kfold_splits",
                        help="K-Fold划分文件目录")
    parser.add_argument("--results-dir", type=str, default="results/kfold_cv_results",
                        help="训练结果保存目录")

    args = parser.parse_args()

    # 检查划分文件是否存在
    if not os.path.exists(args.splits_dir):
        print(f"❌ 错误: 划分文件目录 {args.splits_dir} 不存在")
        print("请先运行: python kfold_cross_validation.py")
        sys.exit(1)

    # 创建训练器
    trainer = KFoldTrainer(
        splits_dir=args.splits_dir,
        results_dir=args.results_dir
    )

    # 简单的配置加载（实际应该用yaml或json）
    config = {
        "dataset": {},
        "save": {}
    }

    # 选择训练模式
    if args.all:
        print("🔄 开始K-Fold交叉验证（所有10个Fold）...")
        trainer.train_all_folds(config, args.gpus)
    elif args.fold is not None:
        if not (0 <= args.fold <= 9):
            print("❌ 错误: Fold编号应该在0-9之间")
            sys.exit(1)
        print(f"🔄 开始训练Fold {args.fold}...")
        trainer.train_fold(args.fold, config, args.gpus)
    else:
        parser.print_help()


if __name__ == "__main__":
    # 如果没有命令行参数，运行演示
    if len(sys.argv) == 1:
        print("📌 K-Fold交叉验证训练演示\n")
        print("使用示例:")
        print("  python kfold_cv_training.py --fold 0          # 训练Fold 0")
        print("  python kfold_cv_training.py --all             # 训练所有Fold")
        print("  python kfold_cv_training.py --all --gpus 0,1  # 使用指定GPU训练\n")

        # 运行演示
        trainer = KFoldTrainer()
        print("✨ 演示：模拟训练所有Fold...")

        # 模拟训练
        config = {"dataset": {}, "save": {}}
        trainer.train_all_folds(config)

        print("\n✓ 演示完成!")
        print(f"结果已保存到: {trainer.results_dir}")
        print("查看详细结果:")
        print(f"  cat {trainer.results_csv}")
        print(f"  cat {os.path.join(trainer.results_dir, 'kfold_summary_report.txt')}")
    else:
        main()
