"""
K-Fold 十折叠交叉验证 - 完整训练指南
根据你的ShipEar项目和LNN模型定制

本指南包括：
1. 数据准备
2. 生成K-Fold划分
3. 配置训练参数
4. 运行完整训练流程
5. 分析结果
"""

# =============================================================================
# 📋 第一步：了解数据结构和位置
# =============================================================================

# 你的项目使用的数据位置 (从configs中看到):
# 
# ShipEar数据集:
#   /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622/
#   ├── train/
#   │   ├── wav/
#   │   │   ├── A/      (class A)
#   │   │   ├── B/      (class B)
#   │   │   ├── C/      (class C)
#   │   │   ├── D/      (class D)
#   │   │   └── E/      (class E)
#   │   ├── mel/
#   │   │   ├── A/, B/, C/, D/, E/
#   │   └── cqt/
#   │       ├── A/, B/, C/, D/, E/
#   │
#   ├── validation/   (结构同train)
#   └── test/         (结构同train)
#
# DeepShip数据集:
#   /media/hdd1/chuxiaohui/AI4Ocean_UATR/DeepShip_622/
#   (结构同上)
#
# ⚠️ 重要：确保数据目录在你能访问的位置

# =============================================================================
# 🚀 第二步：生成K-Fold划分（一次性）
# =============================================================================

"""
【选项A】如果数据在本地或网络访问：

1. 编辑 kfold_cross_validation.py：
   
   def main():
       data_dir = "/media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622"
       # 改成你的实际数据路径

2. 运行脚本生成K-Fold划分：
   
   python kfold_cross_validation.py

3. 输出会在以下位置：
   
   results/kfold_splits/
   ├── kfold_summary.txt      # 汇总
   ├── kfold_fold00.txt       # Fold 0详细信息（包含全部样本路径）
   ├── kfold_fold01.txt
   ├── ...
   ├── kfold_fold09.txt
   └── kfold_indices.txt

4. 查看结果：
   
   cat results/kfold_splits/kfold_summary.txt
   cat results/kfold_splits/kfold_fold00.txt  # 查看样本列表
"""

# =============================================================================
# 📖 第三步：准备K-Fold训练
# =============================================================================

"""
你有两种选择：

【推荐方案】方案A：使用kfold_shipsear_integration.py（一键式）
└─ 自动化程度高
└─ 支持批量训练
└─ 自动统计结果

【灵活方案】方案B：编写自定义训练脚本
└─ 更多控制权
└─ 集成现有训练逻辑
└─ 需要更多代码
"""

# =============================================================================
# 💻 【推荐】方案A：使用ShipEar集成脚本
# =============================================================================

"""
这是最简单的方式，自动化程度最高。

【步骤1】修改集成脚本中的配置

编辑 kfold_shipsear_integration.py，在 main() 中看到：

    parser.add_argument("--data-dir", type=str, default="./data",
                        help="数据目录 (default: ./data)")
    parser.add_argument("--base-config", type=str,
                        default="configs/train_distillation_shipsear.yaml",
                        help="基础配置文件")

你也可以直接在命令行指定：

    python kfold_shipsear_integration.py --setup \\
        --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622

【步骤2】生成K-Fold划分

    python kfold_shipsear_integration.py --setup \\
        --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622

输出示例：
    📊 生成K-Fold划分...
    ✓ 扫描完成: train - 共 1000 个样本
    🔄 划分数据...
    💾 保存划分结果...
    ✓ K-Fold划分完成！

【步骤3】测试单个Fold（可选，建议做）

    python kfold_shipsear_integration.py --train-fold 0 \\
        --gpus 4,5,6,7

这会运行 train_distillation_shipsear.py 进行Fold 0的训练。
观察是否有错误，以及性能如何。

【步骤4】批量训练所有Fold

    python kfold_shipsear_integration.py --train-all \\
        --gpus 4,5,6,7

这会自动运行10个Fold的训练（可能需要数小时）。

【步骤5】查看结果

    python kfold_shipsear_integration.py --results

查看详细的CSV文件：
    
    cat results/kfold_cv_shipsear/kfold_shipsear_results.csv
"""

# =============================================================================
# 🔧 【灵活】方案B：编写自定义训练脚本
# =============================================================================

"""
如果你想更多控制，可以编写自己的训练脚本。

【步骤1】编辑base_config，准备多个折叠配置

# 伪代码示例

import yaml
import os
from kfold_data_loader import KFoldDataLoader

def create_fold_configs(base_config_path, output_dir):
    '''为每个Fold创建独立的config'''
    
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    for fold_idx in range(10):
        # 加载该Fold的数据
        loader = KFoldDataLoader("results/kfold_splits/", fold_idx)
        
        # 修改config
        config = base_config.copy()
        config['dataset']['fold'] = fold_idx
        config['dataset']['total_folds'] = 10
        
        # 设置保存目录
        fold_dir = f"{output_dir}/fold_{fold_idx:02d}"
        os.makedirs(fold_dir, exist_ok=True)
        config['save']['save_dir'] = fold_dir
        
        # 保存该Fold的config
        fold_config_path = f"{fold_dir}/config.yaml"
        with open(fold_config_path, 'w') as f:
            yaml.dump(config, f)
        
        print(f"✓ 已创建 Fold {fold_idx} 的config")

【步骤2】运行训练

# 单个Fold
python train_distillation_shipsear.py --config checkpoints/cv_shipsear/fold_00/config.yaml --gpus 4,5,6,7

# 批量运行所有Fold（使用循环或脚本）
for i in {0..9}; do
    python train_distillation_shipsear.py \\
        --config checkpoints/cv_shipsear/fold_$(printf "%02d" $i)/config.yaml \\
        --gpus 4,5,6,7
done

【步骤3】收集结果

# 手动或脚本方式汇总各Fold的best_acc
"""

# =============================================================================
# 📊 完整的命令流程
# =============================================================================

"""
【推荐流程 - 从头到尾】

1️⃣ 【第1次只需做一次】生成K-Fold划分
   
   python kfold_cross_validation.py
   
   或指定数据目录：
   
   python -c "
   from kfold_cross_validation import KFoldCrossValidator
   validator = KFoldCrossValidator(
       data_dir='/media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622',
       output_dir='results/kfold_splits'
   )
   samples = validator.scan_dataset('train')
   validator.generate_folds(samples)
   validator.save_all_splits()
   "

2️⃣ 【验证划分结果】查看生成的文件
   
   cat results/kfold_splits/kfold_summary.txt
   
   确认：
   - 10个Fold都已生成
   - 每个Fold的类别分布是否平衡
   - 样本路径是否正确

3️⃣ 【可选】保存到Git（确保可复现）
   
   git add results/kfold_splits/
   git commit -m "Add K-Fold splits (ShipEar, seed=42, 1000 samples)"

4️⃣ 【运行训练】选择以下一个：

   # 方式A：使用集成脚本（推荐）
   python kfold_shipsear_integration.py --setup \\
       --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622
   
   # 测试Fold 0
   python kfold_shipsear_integration.py --train-fold 0 --gpus 4,5,6,7
   
   # 批量训练所有Fold
   python kfold_shipsear_integration.py --train-all --gpus 4,5,6,7
   
   # 方式B：手动逐个运行
   python train_distillation_shipsear.py \\
       --config checkpoints/cv_shipsear/fold_00/config.yaml --gpus 4,5,6,7

5️⃣ 【查看结果】
   
   # 查看集成脚本的结果
   python kfold_shipsear_integration.py --results
   
   # 或查看CSV
   cat results/kfold_cv_shipsear/kfold_shipsear_results.csv
"""

# =============================================================================
# 🎯 具体的实际命令示例
# =============================================================================

"""
【场景1】第一次设置（推荐）

# 第1步：修改数据目录并生成K-Fold划分
# 编辑 kfold_cross_validation.py:
#   data_dir = "/media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622"

# 运行生成
cd /d/a_Program_Projects/10-LNN/UATR/10-LNN
python kfold_cross_validation.py

# 等待完成...
# 输出: ✨ 完成! 所有文件已保存到: results/kfold_splits

# 第2步：验证
cat results/kfold_splits/kfold_summary.txt

# 第3步：备份到git
git add results/kfold_splits/
git commit -m "Add K-Fold splits"


【场景2】运行一个Fold进行测试

cd /d/a_Program_Projects/10-LNN/UATR/10-LNN

# 使用集成脚本，会自动：
# 1. 加载Fold 0的数据
# 2. 创建Fold 0的config
# 3. 运行训练
python kfold_shipsear_integration.py --train-fold 0 --gpus 4,5,6,7

# 观察输出，确保：
# ✓ 数据加载正确
# ✓ 训练开始
# ✓ 没有路径或模块错误


【场景3】批量训练所有10个Fold

cd /d/a_Program_Projects/10-LNN/UATR/10-LNN

# 运行所有Fold（可能需要很长时间）
python kfold_shipsear_integration.py --train-all --gpus 4,5,6,7

# 或者使用bash脚本并行运行（更快）
for i in {0..9}; do
    echo "Starting Fold $i..."
    python kfold_shipsear_integration.py --train-fold $i --gpus 4,5,6,7 &
done
wait

# 最后查看结果
python kfold_shipsear_integration.py --results


【场景4】查看K-Fold划分中的样本

# 查看Fold 0的所有训练集样本
head -100 results/kfold_splits/kfold_fold00.txt

# 查看特定行数
sed -n '20,50p' results/kfold_splits/kfold_fold00.txt

# 统计样本数量
grep "^[0-9]" results/kfold_splits/kfold_fold00.txt | wc -l
"""

# =============================================================================
# 📈 分析和可视化结果
# =============================================================================

"""
【查看十折叠的结果统计】

# 1. 生成的CSV文件（如果使用集成脚本）
cat results/kfold_cv_shipsear/kfold_shipsear_results.csv

# 输出示例:
fold,n_train,n_val,best_acc,best_epoch,status,timestamp
0,900,100,0.92,150,success,2025-05-02 10:00:00
1,900,100,0.91,145,success,2025-05-02 11:30:00
...

# 2. 用Python分析
import pandas as pd
df = pd.read_csv('results/kfold_cv_shipsear/kfold_shipsear_results.csv')
print("平均精度:", df['best_acc'].mean())
print("精度范围:", f"{df['best_acc'].min():.4f} - {df['best_acc'].max():.4f}")
print("标准差:", df['best_acc'].std())

# 3. 生成的汇总报告
cat results/kfold_cv_shipsear/training_summary.txt

# 输出示例:
【验证集精度统计】
平均精度:  0.9167
最高精度:  0.9400
最低精度:  0.8900
标准差:    0.0153
"""

# =============================================================================
# 🔍 故障排查
# =============================================================================

"""
【问题1】运行集成脚本时，数据路径错误

解决方案：
1. 检查数据目录是否真的存在
   ls /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622/train/wav/

2. 如果目录不存在，修改data_dir参数：
   python kfold_shipsear_integration.py --setup \\
       --data-dir /你/的/真实/路径/ShipsEar

【问题2】训练时出现模块import错误

解决方案：
1. 检查train_distillation_shipsear.py是否存在
   ls train_distillation_shipsear.py

2. 检查所需的包是否安装
   python -c "import torch; import torchaudio; import yaml"

3. 如果缺少依赖，安装：
   pip install torch torchaudio pyyaml scikit-learn

【问题3】GPU内存不足

解决方案：
1. 减少batch_size（在config中）
2. 使用更少的GPU
3. 使用更小的模型

【问题4】划分结果不平衡

解决方案：
1. 检查原始数据是否平衡
   python -c "
   import os, glob
   for cat in ['A', 'B', 'C', 'D', 'E']:
       path = '/media/.../ShipsEar_622/train/wav/{}'.format(cat)
       count = len(glob.glob(os.path.join(path, '*.wav')))
       print(f'{cat}: {count}')
   "

2. 如果原始数据就不平衡，StratifiedKFold会维持这个不平衡比例
   这是符合预期的行为

【问题5】Fold 0训练后，后续Fold无法启动

解决方案：
1. 检查第一个Fold的日志是否有错误
2. 检查checkpoints目录是否可写
3. 检查是否有未释放的GPU内存：
   nvidia-smi
   # 如果还有进程占用，kill它
"""

# =============================================================================
# ✅ 完整的检查清单
# =============================================================================

"""
运行训练前的检查清单：

□ 数据目录
  - 数据目录存在且可访问
  - 包含train/validation/test文件夹
  - 每个文件夹内有wav/mel/cqt子文件夹
  - 每个子文件夹内有类别文件夹(A,B,C,D,E)

□ Python环境
  - Python版本 >= 3.7
  - torch/torchaudio已安装
  - scikit-learn已安装
  - yaml已安装

□ K-Fold脚本
  - kfold_cross_validation.py存在
  - kfold_data_loader.py存在
  - kfold_shipsear_integration.py存在

□ 配置文件
  - configs/train_distillation_shipsear.yaml存在
  - 内部的data_dir配置（可以通过命令行覆盖）

□ 训练脚本
  - train_distillation_shipsear.py存在
  - 支持--config和--gpus参数

□ 计算资源
  - GPU可用（nvidia-smi能看到设备）
  - GPU内存充足（通常需要>10GB每块GPU）
  - 磁盘空间充足（每个Fold可能产生几GB的检查点）

运行训练时的流程：

1️⃣ 生成K-Fold划分（第一次只需做一次）
   python kfold_cross_validation.py

2️⃣ 验证划分结果
   cat results/kfold_splits/kfold_summary.txt

3️⃣ 测试单个Fold
   python kfold_shipsear_integration.py --train-fold 0 --gpus 4,5,6,7

4️⃣ 如果Fold 0成功，运行所有Fold
   python kfold_shipsear_integration.py --train-all --gpus 4,5,6,7

5️⃣ 查看最终结果
   python kfold_shipsear_integration.py --results
"""

# =============================================================================
# 📚 相关命令快速参考
# =============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              K-Fold十折叠交叉验证 - 快速命令参考                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

【第1步】生成K-Fold划分（一次性）
────────────────────────────────────────────────────────────────────────────────
python kfold_cross_validation.py

【第2步】验证划分结果
────────────────────────────────────────────────────────────────────────────────
cat results/kfold_splits/kfold_summary.txt
cat results/kfold_splits/kfold_fold00.txt

【第3步】测试单个Fold
────────────────────────────────────────────────────────────────────────────────
python kfold_shipsear_integration.py --train-fold 0 --gpus 4,5,6,7

【第4步】批量训练所有Fold
────────────────────────────────────────────────────────────────────────────────
python kfold_shipsear_integration.py --train-all --gpus 4,5,6,7

【第5步】查看结果
────────────────────────────────────────────────────────────────────────────────
python kfold_shipsear_integration.py --results
cat results/kfold_cv_shipsear/kfold_shipsear_results.csv

【其他有用命令】
────────────────────────────────────────────────────────────────────────────────
# 查看GPU使用情况
nvidia-smi

# 监控GPU（实时）
watch -n 1 nvidia-smi

# 查看某个Fold的训练日志
tail -f checkpoints/cv_shipsear/fold_00/training.log

# 快速查看K-Fold快速参考
python KFOLD_QUICK_REFERENCE.py

╔══════════════════════════════════════════════════════════════════════════════╗
║                            完整文档在这里                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

KFOLD_SUMMARY.md         - 总体汇总（5分钟）
KFOLD_README.md          - 快速入门（15分钟）
KFOLD_USAGE_GUIDE.md     - 详细参考（30分钟）
""")
