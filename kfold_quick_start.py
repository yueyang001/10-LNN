#!/usr/bin/env python
"""
K-Fold十折叠交叉验证 - 快速启动脚本
一步步引导用户完成整个训练流程

使用方法:
    python kfold_quick_start.py

或直接在Python中：
    from kfold_quick_start import run_interactive_guide
    run_interactive_guide()
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(title):
    """打印标题"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_step(step_num, title):
    """打印步骤标题"""
    print(f"\n📌 第{step_num}步: {title}")
    print("-"*80)


def check_file_exists(path, name="文件"):
    """检查文件是否存在"""
    if os.path.exists(path):
        print(f"  ✓ {name}存在: {path}")
        return True
    else:
        print(f"  ✗ {name}不存在: {path}")
        return False


def step1_check_environment():
    """第1步：检查环境"""
    print_step(1, "检查开发环境")

    print("\n📦 检查必要的Python包...")

    packages = {
        'torch': 'torch',
        'torchaudio': 'torchaudio',
        'sklearn': 'scikit-learn',
        'yaml': 'PyYAML',
        'numpy': 'numpy',
    }

    all_ok = True
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} 未安装")
            all_ok = False

    if not all_ok:
        print("\n⚠️  缺少依赖包，请安装：")
        print("  pip install torch torchaudio scikit-learn pyyaml")
        return False

    print("\n📂 检查项目文件...")
    files_to_check = [
        ('kfold_cross_validation.py', '主程序'),
        ('kfold_data_loader.py', '数据加载器'),
        ('kfold_shipsear_integration.py', 'ShipEar集成'),
        ('train_distillation_shipsear.py', '训练脚本'),
        ('configs/train_distillation_shipsear.yaml', '训练配置'),
    ]

    for file_path, desc in files_to_check:
        check_file_exists(file_path, desc)

    print("\n📊 检查GPU...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            # 提取GPU信息
            for line in result.stdout.split('\n'):
                if 'NVIDIA' in line or 'GPU' in line:
                    print(f"  ✓ {line.strip()}")
        else:
            print("  ⚠️  无法检测GPU")
    except:
        print("  ⚠️  无法检测GPU")

    return True


def step2_setup_data_directory():
    """第2步：配置数据目录"""
    print_step(2, "配置数据目录")

    print("\n📂 数据目录结构应该是:")
    print("""
    /your/data/path/ShipsEar_622/
    ├── train/
    │   ├── wav/
    │   │   ├── A/   (类别 A 的 .wav 文件)
    │   │   ├── B/
    │   │   ├── C/
    │   │   ├── D/
    │   │   └── E/
    │   ├── mel/     (mel 谱图)
    │   └── cqt/     (cqt 谱图)
    ├── validation/  (结构同 train)
    └── test/        (结构同 train)
    """)

    print("\n请输入你的ShipEar数据目录路径:")
    print("  例如: /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622")

    data_dir = input("  输入路径> ").strip()

    if not data_dir:
        data_dir = "./data"
        print(f"  ℹ️  使用默认路径: {data_dir}")

    # 验证数据目录
    if os.path.exists(data_dir):
        train_dir = os.path.join(data_dir, "train", "wav")
        if os.path.exists(train_dir):
            print(f"\n✓ 数据目录验证成功: {data_dir}")
            # 统计样本数
            import glob
            samples = glob.glob(os.path.join(train_dir, "*/*.wav"))
            print(f"  找到 {len(samples)} 个样本")
            return data_dir
        else:
            print(f"\n✗ 目录结构不对，未找到: {train_dir}")
    else:
        print(f"\n✗ 数据目录不存在: {data_dir}")

    return None


def step3_generate_kfold_splits(data_dir):
    """第3步：生成K-Fold划分"""
    print_step(3, "生成K-Fold划分")

    print(f"\n📊 使用数据目录: {data_dir}")
    print("⏳ 正在生成K-Fold划分...")
    print("   (根据数据量，这可能需要几秒到几分钟)\n")

    # 运行K-Fold生成脚本
    cmd = f"""python -c "
from kfold_cross_validation import KFoldCrossValidator
import os

validator = KFoldCrossValidator(
    data_dir='{data_dir}',
    output_dir='results/kfold_splits',
    seed=42
)

# 扫描训练集
samples = validator.scan_dataset('train')
if not samples:
    print('❌ 无法加载数据')
    exit(1)

# 生成折叠
validator.generate_folds(samples)

# 保存结果
validator.save_all_splits()
validator.save_split_indices()

print('\\n✨ K-Fold划分生成完成!')
"
    """

    result = subprocess.run(cmd, shell=True)

    if result.returncode == 0:
        # 验证输出
        summary_file = "results/kfold_splits/kfold_summary.txt"
        if os.path.exists(summary_file):
            print(f"\n✓ 划分结果已保存到: results/kfold_splits/")
            return True

    print("\n✗ 生成K-Fold划分失败")
    return False


def step4_verify_splits():
    """第4步：验证划分结果"""
    print_step(4, "验证划分结果")

    summary_file = "results/kfold_splits/kfold_summary.txt"

    if not os.path.exists(summary_file):
        print("✗ 未找到划分文件")
        return False

    print(f"\n📄 读取汇总文件: {summary_file}\n")

    with open(summary_file, 'r', encoding='utf-8') as f:
        # 显示前50行
        lines = f.readlines()[:50]
        for line in lines:
            print(line.rstrip())

    print("\n✓ 划分结果验证完成")
    print("\n可以查看详细的Fold信息:")
    print("  cat results/kfold_splits/kfold_fold00.txt  # Fold 0")
    print("  cat results/kfold_splits/kfold_fold01.txt  # Fold 1")

    return True


def step5_test_single_fold():
    """第5步：测试单个Fold"""
    print_step(5, "测试单个Fold训练")

    print("\n现在测试训练一个Fold（Fold 0）来验证整个流程")
    print("这一步可以帮助发现潜在的配置或数据问题")

    response = input("\n是否现在运行Fold 0的测试训练? (y/n)> ").strip().lower()

    if response != 'y':
        print("跳过单个Fold测试")
        return True

    print("\n⏳ 启动Fold 0的测试训练...")
    print("   (这可能需要几分钟到几小时，取决于数据量和硬件)\n")

    cmd = (
        "python kfold_shipsear_integration.py "
        "--train-fold 0 "
        "--gpus 4,5,6,7 "
        "--splits-dir results/kfold_splits"
    )

    print(f"运行命令: {cmd}\n")
    result = subprocess.run(cmd, shell=True)

    if result.returncode == 0:
        print("\n✓ Fold 0 测试训练成功!")
        return True
    else:
        print("\n✗ Fold 0 测试训练失败")
        print("请检查上面的错误信息")
        return False


def step6_run_all_folds():
    """第6步：运行所有Fold"""
    print_step(6, "批量训练所有Fold")

    print("\n现在可以批量训练所有10个Fold")
    print("警告: 这可能需要很长时间（从几小时到一两天）")

    response = input("\n是否现在运行所有Fold的训练? (y/n)> ").strip().lower()

    if response != 'y':
        print("跳过批量训练")
        return True

    print("\n⏳ 启动所有Fold的训练...")
    print("   监控进度: nvidia-smi\n")

    cmd = (
        "python kfold_shipsear_integration.py "
        "--train-all "
        "--gpus 4,5,6,7 "
        "--splits-dir results/kfold_splits"
    )

    print(f"运行命令: {cmd}\n")
    result = subprocess.run(cmd, shell=True)

    if result.returncode == 0:
        print("\n✓ 所有Fold训练完成!")
        return True
    else:
        print("\n✗ 部分或全部Fold训练失败")
        print("请检查错误信息或日志")
        return False


def step7_analyze_results():
    """第7步：分析结果"""
    print_step(7, "分析训练结果")

    results_csv = "results/kfold_cv_shipsear/kfold_shipsear_results.csv"

    if os.path.exists(results_csv):
        print(f"\n📊 读取结果文件: {results_csv}\n")

        with open(results_csv, 'r') as f:
            print(f.read())

        # 尝试计算统计信息
        try:
            import pandas as pd
            df = pd.read_csv(results_csv)

            if 'best_acc' in df.columns:
                accs = df['best_acc'].dropna()
                if len(accs) > 0:
                    print("\n【精度统计】")
                    print(f"  平均精度: {accs.mean():.4f}")
                    print(f"  最高精度: {accs.max():.4f}")
                    print(f"  最低精度: {accs.min():.4f}")
                    print(f"  标准差:   {accs.std():.4f}")
        except:
            pass
    else:
        print("✗ 未找到结果文件")
        print("请确保已完成训练步骤")

    return True


def run_interactive_guide():
    """交互式指南主函数"""
    print_header("K-Fold十折叠交叉验证 - 快速启动指南")

    print("""
欢迎使用K-Fold十折叠交叉验证工具！

本指南会引导你完成以下步骤：
  1. 检查环境
  2. 配置数据目录
  3. 生成K-Fold划分
  4. 验证划分结果
  5. 测试单个Fold
  6. 批量训练所有Fold
  7. 分析训练结果

按Enter继续...
    """)

    input()

    # 第1步：检查环境
    if not step1_check_environment():
        print("\n❌ 环境检查失败，请安装缺失的包后重试")
        return

    # 第2步：配置数据目录
    data_dir = step2_setup_data_directory()
    if not data_dir:
        print("\n❌ 数据目录配置失败")
        return

    # 第3步：生成K-Fold划分
    if not step3_generate_kfold_splits(data_dir):
        print("\n❌ K-Fold划分生成失败")
        return

    # 第4步：验证划分结果
    if not step4_verify_splits():
        print("\n❌ 划分结果验证失败")
        return

    # 第5步：测试单个Fold
    if not step5_test_single_fold():
        print("\n⚠️  单个Fold测试失败，请检查配置后重试")

    # 第6步：运行所有Fold
    if not step6_run_all_folds():
        print("\n⚠️  全部Fold训练失败")

    # 第7步：分析结果
    step7_analyze_results()

    print_header("完成！")
    print("""
✨ K-Fold十折叠交叉验证流程完成！

📊 后续操作:

1. 查看详细结果:
   cat results/kfold_cv_shipsear/training_summary.txt
   cat results/kfold_cv_shipsear/kfold_shipsear_results.csv

2. 保存到版本控制:
   git add results/kfold_splits/
   git add results/kfold_cv_shipsear/
   git commit -m "Add K-Fold cross-validation results"

3. 更多文档:
   - KFOLD_SUMMARY.md        (总体汇总)
   - KFOLD_README.md         (快速入门)
   - KFOLD_USAGE_GUIDE.md    (详细参考)
    """)


def main():
    """主函数"""
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help':
            print(__doc__)
            return

    run_interactive_guide()


if __name__ == "__main__":
    main()
