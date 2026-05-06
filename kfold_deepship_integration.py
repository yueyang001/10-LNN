"""
DeepShip K-Fold交叉验证 + 蒸馏模型训练 完整集成脚本

此脚本演示如何:
1. 加载K-Fold划分结果
2. 为每个Fold运行一次完整的蒸馏训练（学生网络+两种蒸馏方法）
3. 收集所有Fold的结果进行统计分析

使用方式:
    python kfold_deepship_integration.py --fold 0 --config configs/train_LNN_deepship.yaml
    或
    python kfold_deepship_integration.py --all  # 运行所有10个Fold
"""

import os
import sys
import argparse
import csv
import json
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from kfold_data_loader import KFoldDataLoader, load_kfold_splits


class DeepShipKFoldTrainer:
    """DeepShip K-Fold交叉验证训练器"""

    def __init__(self, splits_dir="results/kfold_splits", results_dir="results/kfold_cv_deepship"):
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

    def train_fold(self, fold_idx, config, gpus="4,5,6,7"):
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
        print(f"DeepShip K-Fold Training - Fold {fold_idx}/9")
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

        # 调用真实的训练脚本
        import subprocess
        import time
        start_time = time.time()
        result = self._real_train(fold_idx, fold_save_dir, gpus)
        training_time = time.time() - start_time

        # 如果训练成功，记录时间；如果失败则返回None
        if result:
            result["training_time"] = training_time

        # 保存结果到CSV
        self._save_result_to_csv(fold_idx, len(train_samples), len(val_samples), result)

        return result

    def _real_train(self, fold_idx, save_dir, gpus="4,5,6,7"):
        """调用真实的训练脚本 (蒸馏训练)"""
        import subprocess
        import yaml
        import tempfile

        config_file = "configs/train_distillation_deepship.yaml"
        if not os.path.exists(config_file):
            print(f"⚠️  配置文件 {config_file} 不存在")
            return None

        # 读取配置文件并修改
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        config['save']['save_dir'] = save_dir

        # 适配不同的配置格式：train -> training
        if 'train' in config and 'training' not in config:
            config['training'] = config.pop('train')

        # 适配不同的epoch字段名：epochs -> num_epochs
        if 'training' in config:
            if 'epochs' in config['training'] and 'num_epochs' not in config['training']:
                config['training']['num_epochs'] = config['training'].pop('epochs')
            config['training']['num_epochs'] = 200

        # 修改端口以避免冲突 - 根据不同group使用不同端口
        if 'distributed' not in config:
            config['distributed'] = {}
        config['distributed']['master_addr'] = 'localhost'
        
        # 根据results_dir确定使用哪个端口组
        if 'group3' in save_dir:
            config['distributed']['master_port'] = '12363'  # group3使用12363端口
        elif 'group2' in save_dir:
            config['distributed']['master_port'] = '12362'  # group2使用12362端口
        else:
            config['distributed']['master_port'] = '12361'  # group1使用12361端口

        # 创建临时配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp_config = tmp.name

        try:
            log_file = os.path.join(save_dir, "training.log")
            cmd = (
                f"python train_distillation_shipsear.py "
                f"--config {tmp_config} "
                f"--gpus {gpus} "
                f"2>&1 | tee {log_file}"
            )

            print(f"\n📝 运行蒸馏训练命令: {cmd}\n")
            result = subprocess.run(cmd, shell=True, capture_output=False)

            if result.returncode == 0:
                print(f"✓ Fold {fold_idx} 蒸馏训练成功")
                # 从日志文件中解析真实结果
                parsed_result = self._parse_training_log(log_file)
                if parsed_result:
                    parsed_result["status"] = "success"
                    return parsed_result
                else:
                    print(f"⚠️  无法从日志文件解析结果，使用默认值")
                    return {
                        "best_train_acc": 0.0,
                        "best_val_acc": 0.0,
                        "best_epoch": 200,
                        "status": "success"
                    }
            else:
                print(f"✗ Fold {fold_idx} 蒸馏训练失败 (返回码: {result.returncode})")
                return None
        finally:
            # 删除临时文件
            if os.path.exists(tmp_config):
                os.remove(tmp_config)

    def _parse_training_log(self, log_file):
        """从日志文件中解析训练结果"""
        best_val_acc = 0.0
        best_train_acc = 0.0
        best_epoch = 0

        if not os.path.exists(log_file):
            print(f"⚠️  日志文件不存在: {log_file}")
            return None

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 从末尾开始查找"Training finished"行
            for line in reversed(lines):
                # 查找最后的训练完成信息: "Training finished! Best accuracy: XX.XX%"
                if "Training finished! Best accuracy:" in line:
                    match = re.search(r'Best accuracy: ([\d.]+)%', line)
                    if match:
                        best_val_acc = float(match.group(1))
                        break

            # 查找最佳精度对应的epoch和训练精度
            # 日志格式: "Epoch [X/200] Train Loss: X.XXXX Train Acc: XX.XX% Val Acc: XX.XX%"
            best_found = False
            for line in lines:
                if f"Val Acc: {best_val_acc:.2f}%" in line or (best_val_acc > 0 and f"Val Acc: {best_val_acc:.1f}%" in line):
                    # 提取epoch和训练精度
                    epoch_match = re.search(r'Epoch \[(\d+)/\d+\]', line)
                    train_acc_match = re.search(r'Train Acc: ([\d.]+)%', line)

                    if epoch_match and train_acc_match:
                        best_epoch = int(epoch_match.group(1))
                        best_train_acc = float(train_acc_match.group(1))
                        best_found = True
                        break

            if not best_found and best_val_acc > 0:
                # 如果没找到完全匹配的epoch，找最接近的
                closest_line = None
                for line in lines:
                    if "Val Acc:" in line:
                        val_match = re.search(r'Val Acc: ([\d.]+)%', line)
                        if val_match:
                            val_acc = float(val_match.group(1))
                            if abs(val_acc - best_val_acc) < 0.1:
                                closest_line = line

                if closest_line:
                    epoch_match = re.search(r'Epoch \[(\d+)/\d+\]', closest_line)
                    train_acc_match = re.search(r'Train Acc: ([\d.]+)%', closest_line)
                    if epoch_match and train_acc_match:
                        best_epoch = int(epoch_match.group(1))
                        best_train_acc = float(train_acc_match.group(1))

            return {
                "best_train_acc": best_train_acc,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch
            }
        except Exception as e:
            print(f"⚠️  解析日志文件失败: {e}")
            return None

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

    def train_all_folds(self, config, gpus="4,5,6,7"):
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
            f.write("DeepShip K-Fold交叉验证 - 训练总结报告\n")
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
    parser = argparse.ArgumentParser(description="DeepShip K-Fold交叉验证训练")
    parser.add_argument("--fold", type=int, default=None, help="训练指定的Fold (0-9)")
    parser.add_argument("--all", action="store_true", help="训练所有Fold")
    parser.add_argument("--config", type=str, default="configs/train_distillation_deepship.yaml",
                        help="训练配置文件路径")
    parser.add_argument("--gpus", type=str, default="4,5,6,7", help="GPU编号")
    parser.add_argument("--splits-dir", type=str, default="results/kfold_splits",
                        help="K-Fold划分文件目录")
    parser.add_argument("--results-dir", type=str, default="results/kfold_cv_deepship",
                        help="训练结果保存目录")

    args = parser.parse_args()

    # 检查划分文件是否存在
    if not os.path.exists(args.splits_dir):
        print(f"❌ 错误: 划分文件目录 {args.splits_dir} 不存在")
        print("请先运行: python kfold_cross_validation.py")
        sys.exit(1)

    # 创建训练器
    trainer = DeepShipKFoldTrainer(
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
        print("🔄 开始DeepShip K-Fold交叉验证（所有10个Fold）...")
        trainer.train_all_folds(config, args.gpus)
    elif args.fold is not None:
        if not (0 <= args.fold <= 9):
            print("❌ 错误: Fold编号应该在0-9之间")
            sys.exit(1)
        print(f"🔄 开始训练DeepShip Fold {args.fold}...")
        trainer.train_fold(args.fold, config, args.gpus)
    else:
        parser.print_help()


if __name__ == "__main__":
    # 如果没有命令行参数，运行演示
    if len(sys.argv) == 1:
        print("📌 DeepShip K-Fold交叉验证 + 蒸馏训练演示\n")
        print("使用示例:")
        print("  python kfold_deepship_integration.py --fold 0          # 训练Fold 0")
        print("  python kfold_deepship_integration.py --all             # 训练所有Fold")
        print("  python kfold_deepship_integration.py --all --gpus 0,1  # 使用指定GPU训练\n")

        # 运行演示
        trainer = DeepShipKFoldTrainer()
        print("✨ 演示：模拟训练所有Fold...\n")

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
