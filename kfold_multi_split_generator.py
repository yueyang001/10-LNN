"""
多组K-Fold划分生成器

生成多个不同的K-Fold划分（使用不同的随机种子），
便于对比选择更好的划分用于训练。

使用方法:
    python kfold_multi_split_generator.py

或指定参数:
    python kfold_multi_split_generator.py \
        --data-dir /path/to/data \
        --num-splits 3 \
        --output-dir results/kfold_splits_multi
"""

import os
import argparse
import json
from datetime import datetime
from pathlib import Path
from kfold_cross_validation import KFoldCrossValidator


class MultiSplitGenerator:
    """多组K-Fold划分生成器"""

    def __init__(self, data_dir, output_dir="results/kfold_splits_multi", num_splits=3):
        """
        参数:
            data_dir: 数据目录
            output_dir: 输出目录
            num_splits: 生成的划分组数
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.num_splits = num_splits
        self.split_configs = []

        os.makedirs(output_dir, exist_ok=True)

    def generate_multiple_splits(self, seeds=None):
        """
        生成多组K-Fold划分

        参数:
            seeds: 随机种子列表。如果为None，自动生成
        """
        if seeds is None:
            seeds = [42, 123, 456][:self.num_splits]

        print(f"\n📊 生成 {len(seeds)} 组K-Fold划分")
        print(f"   使用的随机种子: {seeds}\n")

        results = {}

        for split_id, seed in enumerate(seeds):
            print(f"\n{'='*80}")
            print(f"【第 {split_id+1}/{len(seeds)} 组】随机种子: {seed}")
            print(f"{'='*80}")

            # 创建输出目录
            split_name = f"split_{split_id:02d}_seed{seed}"
            split_dir = os.path.join(self.output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)

            # 生成该组划分
            validator = KFoldCrossValidator(
                data_dir=self.data_dir,
                output_dir=split_dir,
                seed=seed
            )

            # 扫描数据
            samples = validator.scan_dataset("train")
            if not samples:
                print(f"❌ 无法加载数据")
                continue

            # 生成折叠
            print(f"\n🔄 生成十折叠划分...")
            validator.generate_folds(samples)

            # 保存结果
            print(f"\n💾 保存划分结果...")
            validator.save_all_splits(suffix="")
            validator.save_split_indices(suffix="")

            # 记录配置
            config_info = {
                "split_id": split_id,
                "split_name": split_name,
                "seed": seed,
                "data_dir": self.data_dir,
                "output_dir": split_dir,
                "n_samples": len(samples),
                "n_folds": 10,
                "generated_time": datetime.now().isoformat(),
            }

            self.split_configs.append(config_info)
            results[split_name] = config_info

            print(f"✓ 完成: {split_name}")

        # 生成汇总配置
        self._save_summary_config(results)

        return results

    def _save_summary_config(self, results):
        """保存所有划分的汇总配置"""
        summary_file = os.path.join(self.output_dir, "splits_summary.json")

        summary_data = {
            "total_splits": len(results),
            "output_dir": self.output_dir,
            "generated_time": datetime.now().isoformat(),
            "splits": results
        }

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 汇总配置已保存: {summary_file}")

    def generate_comparison_report(self):
        """生成对比报告"""
        report_file = os.path.join(self.output_dir, "splits_comparison.txt")

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("多组K-Fold划分对比报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            f.write("【划分概览】\n")
            f.write(f"总划分组数: {len(self.split_configs)}\n")
            f.write(f"输出目录: {self.output_dir}\n\n")

            f.write("【各组详情】\n")
            for config in self.split_configs:
                f.write(f"\n{config['split_name']}\n")
                f.write(f"  随机种子: {config['seed']}\n")
                f.write(f"  样本数: {config['n_samples']}\n")
                f.write(f"  数据目录: {config['data_dir']}\n")
                f.write(f"  生成时间: {config['generated_time']}\n")
                f.write(f"  详细文件:\n")
                f.write(f"    - {config['split_name']}/kfold_summary.txt\n")
                f.write(f"    - {config['split_name']}/kfold_fold*.txt\n")

            f.write("\n" + "="*80 + "\n")
            f.write("【使用方法】\n")
            f.write("""
选择一组划分进行训练:

  # 使用split_00 (seed=42)
  python kfold_shipsear_integration.py \\
      --train-all \\
      --gpus 4,5,6,7 \\
      --splits-dir results/kfold_splits_multi/split_00_seed42

  # 使用split_01 (seed=123)
  python kfold_shipsear_integration.py \\
      --train-all \\
      --gpus 4,5,6,7 \\
      --splits-dir results/kfold_splits_multi/split_01_seed123

  # 对比结果
  cat results/kfold_splits_multi/splits_comparison.txt
""")

        print(f"✓ 对比报告已保存: {report_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="生成多组K-Fold划分",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 生成3组K-Fold划分（使用默认种子）
  python kfold_multi_split_generator.py \\
      --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622

  # 生成5组K-Fold划分（自定义种子）
  python kfold_multi_split_generator.py \\
      --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622 \\
      --num-splits 5 \\
      --seeds 42 123 456 789 999
        """
    )

    parser.add_argument("--data-dir", type=str, default="./data",
                        help="数据目录")
    parser.add_argument("--output-dir", type=str, default="results/kfold_splits_multi",
                        help="输出目录")
    parser.add_argument("--num-splits", type=int, default=3,
                        help="生成的划分组数 (default: 3)")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="指定随机种子列表（可选）")

    args = parser.parse_args()

    # 验证数据目录
    if not os.path.exists(args.data_dir):
        print(f"❌ 数据目录不存在: {args.data_dir}")
        return

    train_dir = os.path.join(args.data_dir, "train", "wav")
    if not os.path.exists(train_dir):
        print(f"❌ 数据目录结构不对: {train_dir}")
        return

    # 创建生成器
    generator = MultiSplitGenerator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_splits=args.num_splits
    )

    # 生成划分
    results = generator.generate_multiple_splits(seeds=args.seeds)

    # 生成报告
    generator.generate_comparison_report()

    print("\n" + "="*80)
    print("✨ 多组K-Fold划分生成完成!")
    print("="*80)
    print(f"\n输出目录: {args.output_dir}")
    print("\n生成的划分:")
    for split_name in results.keys():
        print(f"  - {split_name}/")

    print("\n📊 对比报告: " + os.path.join(args.output_dir, "splits_comparison.txt"))
    print("\n下一步:")
    print("  1. 选择一个划分进行训练")
    print("  2. 比较不同划分的训练结果")
    print("  3. 选择性能最好的划分")


if __name__ == "__main__":
    main()
