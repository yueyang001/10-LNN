"""
K-Fold划分结果对比分析工具

用于对比不同种子生成的K-Fold划分的训练结果，
帮助选择性能最好的划分。

使用方法:
    python kfold_comparison_analyzer.py \\
        --results-dir results/kfold_cv_shipsear_multi
"""

import os
import csv
import json
import argparse
from datetime import datetime
from pathlib import Path
import statistics


class ComparisonAnalyzer:
    """划分结果对比分析器"""

    def __init__(self, results_dir):
        """
        参数:
            results_dir: 包含多个split结果的目录
        """
        self.results_dir = results_dir
        self.splits_results = {}

    def load_split_results(self):
        """加载所有split的结果"""
        if not os.path.exists(self.results_dir):
            print(f"❌ 结果目录不存在: {self.results_dir}")
            return False

        # 查找所有split_*的目录
        for split_dir in os.listdir(self.results_dir):
            split_path = os.path.join(self.results_dir, split_dir)
            if not os.path.isdir(split_path):
                continue

            if not split_dir.startswith("split_"):
                continue

            csv_file = os.path.join(split_path, "kfold_shipsear_results.csv")
            if os.path.exists(csv_file):
                self.splits_results[split_dir] = self._load_csv(csv_file)
                print(f"✓ 已加载: {split_dir}")
            else:
                print(f"⚠️  未找到结果文件: {split_dir}")

        return len(self.splits_results) > 0

    def _load_csv(self, csv_file):
        """加载CSV文件"""
        results = []
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        row['best_acc'] = float(row['best_acc']) if row['best_acc'] else None
                        results.append(row)
                    except:
                        pass
        except Exception as e:
            print(f"  ✗ 读取失败: {e}")
        return results

    def analyze_and_compare(self):
        """分析和对比所有划分的结果"""
        if not self.splits_results:
            print("❌ 未加载到任何结果")
            return

        print("\n" + "="*80)
        print("K-Fold划分结果对比分析")
        print("="*80 + "\n")

        analysis = {}

        for split_name in sorted(self.splits_results.keys()):
            results = self.splits_results[split_name]
            accs = [r['best_acc'] for r in results if r['best_acc'] is not None]

            if len(accs) > 0:
                analysis[split_name] = {
                    "count": len(accs),
                    "mean": statistics.mean(accs),
                    "median": statistics.median(accs),
                    "stdev": statistics.stdev(accs) if len(accs) > 1 else 0,
                    "min": min(accs),
                    "max": max(accs),
                    "all": accs
                }

        # 按平均精度排序
        sorted_splits = sorted(analysis.items(), key=lambda x: x[1]['mean'], reverse=True)

        # 打印对比表格
        print("【各划分性能统计】\n")
        print(f"{'排名':<6} {'划分名称':<25} {'平均精度':<12} {'中位数':<12} {'标准差':<12} {'范围':<20}")
        print("-"*95)

        for rank, (split_name, stats) in enumerate(sorted_splits, 1):
            avg = stats['mean']
            median = stats['median']
            stdev = stats['stdev']
            range_str = f"[{stats['min']:.4f}, {stats['max']:.4f}]"

            # 用★标记最好的
            marker = " ⭐" if rank == 1 else ""
            print(f"{rank:<6} {split_name:<25} {avg:.4f}       {median:.4f}       {stdev:.4f}       {range_str:<20}{marker}")

        # 详细对比
        print("\n" + "="*80)
        print("【详细精度对比】\n")

        for rank, (split_name, stats) in enumerate(sorted_splits, 1):
            print(f"{rank}. {split_name}")
            print(f"   平均精度: {stats['mean']:.4f}")
            print(f"   中位数:   {stats['median']:.4f}")
            print(f"   标准差:   {stats['stdev']:.4f}")
            print(f"   最高精度: {stats['max']:.4f}")
            print(f"   最低精度: {stats['min']:.4f}")
            print(f"   Fold精度列表: {', '.join([f'{acc:.4f}' for acc in sorted(stats['all'])])}")
            print()

        # 统计分析
        print("="*80)
        print("【统计分析】\n")

        all_means = [stats['mean'] for stats in analysis.values()]
        print(f"所有划分的平均精度均值: {statistics.mean(all_means):.4f}")
        print(f"所有划分的平均精度中位数: {statistics.median(all_means):.4f}")
        if len(all_means) > 1:
            print(f"所有划分的平均精度标准差: {statistics.stdev(all_means):.4f}")
        print(f"最好的划分: {sorted_splits[0][0]} (平均精度: {sorted_splits[0][1]['mean']:.4f})")
        print(f"最差的划分: {sorted_splits[-1][0]} (平均精度: {sorted_splits[-1][1]['mean']:.4f})")

        # 生成推荐
        print("\n" + "="*80)
        print("【推荐】\n")
        best_split = sorted_splits[0][0]
        best_mean = sorted_splits[0][1]['mean']
        best_stdev = sorted_splits[0][1]['stdev']

        print(f"✓ 推荐使用: {best_split}")
        print(f"  理由:")
        print(f"  - 平均精度最高: {best_mean:.4f}")
        print(f"  - 标准差: {best_stdev:.4f}")
        print(f"  - 性能稳定，Fold间差异小")

        # 返回最佳划分
        return best_split, analysis

    def generate_report(self, output_file=None):
        """生成对比报告"""
        if not self.splits_results:
            print("❌ 未加载到任何结果")
            return

        best_split, analysis = self.analyze_and_compare()

        if output_file is None:
            output_file = os.path.join(self.results_dir, "comparison_report.txt")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("K-Fold划分结果对比报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            f.write("【各划分性能统计】\n\n")

            sorted_splits = sorted(analysis.items(), key=lambda x: x[1]['mean'], reverse=True)

            for rank, (split_name, stats) in enumerate(sorted_splits, 1):
                marker = "⭐" if rank == 1 else ""
                f.write(f"{rank}. {split_name} {marker}\n")
                f.write(f"   平均精度: {stats['mean']:.4f}\n")
                f.write(f"   中位数:   {stats['median']:.4f}\n")
                f.write(f"   标准差:   {stats['stdev']:.4f}\n")
                f.write(f"   精度范围: [{stats['min']:.4f}, {stats['max']:.4f}]\n")
                f.write(f"   各Fold精度: {', '.join([f'{acc:.4f}' for acc in sorted(stats['all'])])}\n\n")

            f.write("\n" + "="*80 + "\n")
            f.write("【推荐】\n\n")
            best_mean = sorted_splits[0][1]['mean']
            best_stdev = sorted_splits[0][1]['stdev']
            f.write(f"✓ 推荐使用: {best_split}\n")
            f.write(f"  平均精度: {best_mean:.4f}\n")
            f.write(f"  标准差: {best_stdev:.4f}\n")

        print(f"\n✓ 报告已保存: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="K-Fold划分结果对比分析")
    parser.add_argument("--results-dir", type=str, default="results/kfold_cv_shipsear_multi",
                        help="结果目录")
    parser.add_argument("--output", type=str, help="输出报告文件")

    args = parser.parse_args()

    # 创建分析器
    analyzer = ComparisonAnalyzer(args.results_dir)

    # 加载结果
    print("📊 加载划分结果...\n")
    if not analyzer.load_split_results():
        print("\n❌ 无法加载任何结果")
        print("请确保已完成多组K-Fold的训练")
        return

    # 分析对比
    print()
    best_split, _ = analyzer.analyze_and_compare()

    # 生成报告
    analyzer.generate_report(args.output)

    print("\n" + "="*80)
    print("✨ 对比分析完成!")
    print("="*80)
    print(f"\n🎯 最佳划分: {best_split}")
    print("\n后续步骤:")
    print("  1. 使用最佳划分重新训练")
    print("  2. 保存最优结果")
    print("  3. 在论文中记录使用的划分和性能")


if __name__ == "__main__":
    main()
