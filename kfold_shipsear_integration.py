"""
K-Fold 50折叠交叉验证 + ShipEar实际集成脚本

此脚本展示如何将K-Fold与现有的ShipEar训练系统集成。
它可以:
1. 生成K-Fold划分
2. 为每个Fold创建对应的config
3. 批量运行训练
4. 收集结果统计

使用示例:
    # 第一次运行：生成K-Fold划分
    python kfold_shipsear_integration.py --setup --data-dir ./data

    # 然后运行训练
    python kfold_shipsear_integration.py --train-all --gpus 4,5,6,7

    # 或者只训练某个Fold
    python kfold_shipsear_integration.py --train-fold 0 --gpus 4,5,6,7

    # 查看结果
    python kfold_shipsear_integration.py --results
"""

import os
import sys
import yaml
import argparse
import subprocess
import csv
from datetime import datetime
from pathlib import Path
from kfold_data_loader import KFoldDataLoader

# 动态添加项目路径以导入KFoldCrossValidator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments', 'cv'))
try:
    from experiments.cv.kfold_cross_validation import KFoldCrossValidator
except ImportError:
    print("❌ 警告: 无法导入 KFoldCrossValidator，请确保 experiments/cv/kfold_cross_validation.py 存在")
    KFoldCrossValidator = None


class ShipEarKFoldIntegration:
    """ShipEar + K-Fold集成类"""

    def __init__(self, base_config="configs/train_distillation_shipsear.yaml",
                 data_dir="./data",
                 splits_dir="results/kfold_splits",
                 checkpoints_dir="checkpoints/cv_shipsear",
                 results_dir="results/kfold_cv_shipsear"):
        """
        参数:
            base_config: 基础配置文件路径
            data_dir: 数据目录
            splits_dir: K-Fold划分保存目录
            checkpoints_dir: 模型检查点保存目录
            results_dir: 结果保存目录
        """
        self.base_config = base_config
        self.data_dir = data_dir
        self.splits_dir = splits_dir
        self.checkpoints_dir = checkpoints_dir
        self.results_dir = results_dir

        os.makedirs(splits_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # 初始化结果CSV
        self.results_csv = os.path.join(results_dir, "kfold_shipsear_results.csv")
        if not os.path.exists(self.results_csv):
            with open(self.results_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "fold",
                    "n_train",
                    "n_val",
                    "best_acc",
                    "best_auc",
                    "best_f1",
                    "best_epoch",
                    "status",
                    "timestamp"
                ])

    def setup_kfold_splits(self):
        """生成K-Fold划分"""
        if KFoldCrossValidator is None:
            print("❌ 错误: 无法导入 KFoldCrossValidator")
            return False

        print("\n📊 生成K-Fold划分...")

        validator = KFoldCrossValidator(
            data_dir=self.data_dir,
            output_dir=self.splits_dir
        )

        # 扫描训练数据
        samples = validator.scan_dataset(data_flag="train")
        if not samples:
            print("❌ 无法获取数据")
            return False

        # 生成折叠
        print("\n🔄 划分数据...")
        validator.generate_folds(samples)

        # 保存结果
        print("\n💾 保存划分结果...")
        validator.save_all_splits(suffix="")
        validator.save_split_indices(suffix="")

        print(f"\n✓ K-Fold划分完成！")
        print(f"  汇总: {os.path.join(self.splits_dir, 'kfold_summary.txt')}")
        return True

    def create_fold_config(self, fold_idx, base_config_path):
        """为指定Fold创建配置文件"""
        # 加载基础配置
        with open(base_config_path, "r") as f:
            config = yaml.safe_load(f)

        # 更新数据集信息
        config["dataset"]["fold"] = fold_idx
        config["dataset"]["total_folds"] = 10

        # 更新保存目录
        fold_save_dir = os.path.join(self.checkpoints_dir, f"fold_{fold_idx:02d}")
        os.makedirs(fold_save_dir, exist_ok=True)
        config["save"]["save_dir"] = fold_save_dir

        # 保存临时配置
        fold_config_path = os.path.join(fold_save_dir, "config.yaml")
        with open(fold_config_path, "w") as f:
            yaml.dump(config, f)

        return fold_config_path, fold_save_dir

    def train_fold(self, fold_idx, gpus="4,5,6,7"):
        """训练单个Fold"""
        print(f"\n{'='*80}")
        print(f"训练 Fold {fold_idx}/49")
        print(f"{'='*80}")

        # 检查K-Fold划分文件是否存在
        fold_file = os.path.join(self.splits_dir, f"kfold_fold{fold_idx:02d}.txt")
        if not os.path.exists(fold_file):
            print(f"❌ 未找到Fold文件: {fold_file}")
            print("请先运行: --setup")
            return False

        # 加载该Fold的数据
        loader = KFoldDataLoader(self.splits_dir, fold_idx)
        train_samples = loader.get_train_samples()
        val_samples = loader.get_val_samples()

        print(f"✓ 训练集: {len(train_samples)} 样本")
        print(f"✓ 验证集: {len(val_samples)} 样本")

        # 创建该Fold的配置
        fold_config_path, fold_save_dir = self.create_fold_config(
            fold_idx, self.base_config
        )

        # 构造训练命令
        # 注意：这里假设你的训练脚本是 train_distillation_shipsear.py
        # 如果不同，请修改命令
        cmd = (
            f"python train_distillation_shipsear.py "
            f"--config {fold_config_path} "
            f"--gpus {gpus}"
        )

        print(f"\n🚀 启动训练命令:")
        print(f"  {cmd}")

        # 运行训练
        try:
            result = subprocess.run(cmd, shell=True, cwd="./")
            success = result.returncode == 0
        except Exception as e:
            print(f"❌ 训练出错: {e}")
            success = False

        # 记录结果
        best_acc = None
        best_auc = None
        best_f1 = None
        best_epoch = None

        if success:
            best_model_path = os.path.join(fold_save_dir, "best_student.pth")
            if os.path.exists(best_model_path):
                try:
                    import torch
                    ckpt = torch.load(best_model_path, map_location="cpu")
                    best_acc = ckpt.get("best_acc")
                    best_auc = ckpt.get("best_auc")
                    best_f1 = ckpt.get("best_f1")
                    best_epoch = ckpt.get("epoch")
                except:
                    pass

        # 保存结果到CSV
        with open(self.results_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                fold_idx,
                len(train_samples),
                len(val_samples),
                best_acc,
                best_auc,
                best_f1,
                best_epoch,
                "success" if success else "failed",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])

        status_str = "✓ 成功" if success else "✗ 失败"
        print(f"\n{status_str} Fold {fold_idx} 训练完成")
        if best_acc is not None:
            print(f"  最佳精度: {best_acc:.4f}")
            print(f"  最佳 AUC: {best_auc:.4f}")
            print(f"  最佳 F1: {best_f1:.4f}")
            print(f"  最佳Epoch: {best_epoch}")

        return success

    def train_all_folds(self, gpus="4,5,6,7"):
        """训练50个Fold，分5组进行10折叠交叉验证"""
        print("\n🔄 开始批量训练（分5组进行10折叠交叉验证）...\n")

        results = {}

        # 分组规则: 5个独立的10折叠组
        fold_groups = [
            (0, 10, "Group_0"),   # fold_0 to fold_9
            (10, 20, "Group_1"),  # fold_10 to fold_19
            (20, 30, "Group_2"),  # fold_20 to fold_29
            (30, 40, "Group_3"),  # fold_30 to fold_39
            (40, 50, "Group_4")   # fold_40 to fold_49
        ]

        for start_fold, end_fold, group_name in fold_groups:
            print(f"\n{'='*80}")
            print(f"训练 {group_name} (Fold {start_fold}-{end_fold-1}) 的10折叠交叉验证")
            print(f"{'='*80}")

            group_results = {}
            for fold_idx in range(start_fold, end_fold):
                success = self.train_fold(fold_idx, gpus)
                results[fold_idx] = success
                group_results[fold_idx] = success

            # 为每组生成单独的汇总报告
            self._generate_group_report(group_name, group_results, start_fold, end_fold)

        # 统计成功/失败
        n_success = sum(1 for v in results.values() if v)
        print(f"\n{'='*80}")
        print(f"训练完成统计: {n_success}/50 个Fold成功")
        print(f"{'='*80}")

        # 生成总结报告
        self.generate_report()

        return results

    def _generate_group_report(self, group_name, group_results, start_fold, end_fold):
        """为每个组生成汇总报告"""
        report_file = os.path.join(self.results_dir, f"training_summary_{group_name}.txt")

        n_success = sum(1 for v in group_results.values() if v)

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"ShipEar K-Fold {group_name} (Fold {start_fold}-{end_fold-1}) 10折叠交叉验证 - 训练总结\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            f.write("【训练完成统计】\n")
            f.write(f"成功: {n_success}/10 个Fold\n")
            f.write(f"失败: {10-n_success}/10 个Fold\n\n")

            f.write("【各Fold结果】\n")
            for fold_idx in sorted(group_results.keys()):
                status = "成功" if group_results[fold_idx] else "失败"
                f.write(f"Fold {fold_idx}: {status}\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"✓ {group_name}汇总报告已保存: {report_file}")

    def generate_report(self):
        """生成训练结果报告"""
        report_file = os.path.join(self.results_dir, "training_summary.txt")

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("ShipEar K-Fold 50折叠交叉验证 - 训练总结\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            # 读取结果CSV
            if os.path.exists(self.results_csv):
                f.write("【训练结果汇总】\n")
                import numpy as np

                accs = []
                aucs = []
                f1s = []

                with open(self.results_csv, "r") as csv_f:
                    reader = csv.DictReader(csv_f)
                    rows = list(reader)
                    for row in rows:
                        f.write(
                            f"Fold {row['fold']}: "
                            f"acc={row['best_acc']}, "
                            f"auc={row['best_auc']}, "
                            f"f1={row['best_f1']}, "
                            f"status={row['status']}\n"
                        )

                        # 收集指标用于统计
                        if row['best_acc']:
                            try:
                                accs.append(float(row['best_acc']))
                            except:
                                pass
                        if row['best_auc']:
                            try:
                                aucs.append(float(row['best_auc']))
                            except:
                                pass
                        if row['best_f1']:
                            try:
                                f1s.append(float(row['best_f1']))
                            except:
                                pass

                f.write("\n【指标统计】\n")
                if accs:
                    f.write(f"ACC - 平均: {np.mean(accs):.4f}, 最高: {np.max(accs):.4f}, 最低: {np.min(accs):.4f}, 标准差: {np.std(accs):.4f}\n")
                if aucs:
                    f.write(f"AUC - 平均: {np.mean(aucs):.4f}, 最高: {np.max(aucs):.4f}, 最低: {np.min(aucs):.4f}, 标准差: {np.std(aucs):.4f}\n")
                if f1s:
                    f.write(f"F1  - 平均: {np.mean(f1s):.4f}, 最高: {np.max(f1s):.4f}, 最低: {np.min(f1s):.4f}, 标准差: {np.std(f1s):.4f}\n")

                f.write(f"\n【详细结果CSV】\n")
                f.write(f"查看详细结果: {self.results_csv}\n")

        print(f"✓ 报告已保存: {report_file}")

    def print_results(self):
        """打印训练结果"""
        print("\n📊 K-Fold交叉验证结果")
        print("=" * 80)

        if os.path.exists(self.results_csv):
            with open(self.results_csv, "r") as f:
                print(f.read())
        else:
            print("❌ 未找到结果文件")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="ShipEar K-Fold交叉验证集成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 第1步: 生成K-Fold划分
  python kfold_shipsear_integration.py --setup --data-dir ./data

  # 第2步: 训练所有Fold
  python kfold_shipsear_integration.py --train-all --gpus 4,5,6,7

  # 或只训练某个Fold进行测试
  python kfold_shipsear_integration.py --train-fold 0 --gpus 4,5,6,7

  # 查看结果
  python kfold_shipsear_integration.py --results
        """
    )

    parser.add_argument("--setup", action="store_true",
                        help="生成K-Fold划分")
    parser.add_argument("--train-fold", type=int,
                        help="训练指定的Fold (0-9)")
    parser.add_argument("--train-all", action="store_true",
                        help="训练所有Fold")
    parser.add_argument("--results", action="store_true",
                        help="显示训练结果")

    parser.add_argument("--data-dir", type=str, default="./data",
                        help="数据目录 (default: ./data)")
    parser.add_argument("--base-config", type=str,
                        default="configs/train_distillation_shipsear.yaml",
                        help="基础配置文件")
    parser.add_argument("--gpus", type=str, default="4,5,6,7",
                        help="GPU编号 (default: 4,5,6,7)")
    parser.add_argument("--splits-dir", type=str, default="results/kfold_splits",
                        help="K-Fold划分目录")
    parser.add_argument("--checkpoints-dir", type=str,
                        default="checkpoints/cv_shipsear",
                        help="检查点保存目录")
    parser.add_argument("--results-dir", type=str,
                        default="results/kfold_cv_shipsear",
                        help="结果保存目录")

    args = parser.parse_args()

    # 创建集成工具
    integration = ShipEarKFoldIntegration(
        base_config=args.base_config,
        data_dir=args.data_dir,
        splits_dir=args.splits_dir,
        checkpoints_dir=args.checkpoints_dir,
        results_dir=args.results_dir
    )

    # 执行操作
    if args.setup:
        integration.setup_kfold_splits()

    elif args.train_all:
        integration.train_all_folds(args.gpus)
        integration.generate_report()

    elif args.train_fold is not None:
        if not (0 <= args.train_fold <= 49):
            print("❌ 错误: Fold编号应该在0-49之间")
            sys.exit(1)
        integration.train_fold(args.train_fold, args.gpus)

    elif args.results:
        integration.print_results()

    else:
        parser.print_help()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("📌 ShipEar K-Fold交叉验证集成工具\n")
        print("快速开始:")
        print("  1. 生成K-Fold划分:")
        print("     python kfold_shipsear_integration.py --setup --data-dir ./data\n")
        print("  2. 批量训练:")
        print("     python kfold_shipsear_integration.py --train-all --gpus 4,5,6,7\n")
        print("  3. 查看结果:")
        print("     python kfold_shipsear_integration.py --results\n")
        parser = argparse.ArgumentParser()
        parser.print_help()
    else:
        main()
