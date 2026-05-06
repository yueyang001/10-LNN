# K-Fold 十折叠交叉验证 - ShipEar & DeepShip 训练集成指南

## 目录
1. [快速开始](#快速开始)
2. [核心概念](#核心概念)
3. [两个集成脚本对比](#两个集成脚本对比)
4. [ShipEar集成脚本使用](#shipsear集成脚本使用)
5. [DeepShip集成脚本使用](#deepship集成脚本使用)
6. [完整工作流程](#完整工作流程)
7. [命令参考](#命令参考)
8. [常见问题](#常见问题)
9. [故障排查](#故障排查)

---

## 快速开始

### 30秒快速启动（ShipEar）

```bash
# 1. 生成K-Fold划分（仅需要一次）
python kfold_cross_validation.py

# 2. 验证生成结果
cat results/kfold_splits/kfold_summary.txt

# 3. 测试Fold 0
python kfold_shipsear_integration.py --train-fold 0 --gpus 4,5,6,7

# 4. 训练所有Fold
python kfold_shipsear_integration.py --train-all --gpus 4,5,6,7

# 5. 查看最终结果
python kfold_shipsear_integration.py --results
```

### 30秒快速启动（DeepShip）

```bash
# 1. 生成K-Fold划分（仅需要一次）
python kfold_cross_validation.py

# 2. 验证生成结果
cat results/kfold_splits/kfold_summary.txt

# 3. 测试Fold 0
python kfold_deepship_integration.py --fold 0 --gpus 4,5,6,7

# 4. 训练所有Fold
python kfold_deepship_integration.py --all --gpus 4,5,6,7
```

---

## 核心概念

### 什么是K-Fold交叉验证？
- **十折叠(10-Fold)**：数据被随机分成10个等大小的子集
- **交叉验证**：每次用9个子集训练，1个子集测试，共进行10次
- **优势**：充分利用有限数据，得到更稳健的性能评估

### 系统结构

```
┌─────────────────────────────────────────────────────┐
│          K-Fold 十折交叉验证系统                      │
├─────────────────────────────────────────────────────┤
│                                                       │
│  1. 数据划分层(Data Splitting)                       │
│     └─ kfold_cross_validation.py                    │
│        ↓ 生成10个平衡的Fold                          │
│        └─ results/kfold_splits/*.txt                │
│                                                       │
│  2. 数据加载层(Data Loading)                         │
│     └─ kfold_data_loader.py                         │
│        ↓ Python API 加载Fold数据                    │
│        └─ KFoldDataLoader 类                        │
│                                                       │
│  3a. ShipEar训练执行层(Training Execution)           │
│     └─ kfold_shipsear_integration.py                │
│        ↓ 执行单个或批量Fold训练                     │
│        └─ results/kfold_cv_shipsear/                │
│                                                       │
│  3b. DeepShip训练执行层(Training Execution)          │
│     └─ kfold_deepship_integration.py                │
│        ↓ 执行单个或批量Fold蒸馏训练                  │
│        └─ results/kfold_cv_deepship/                │
│                                                       │
│  4. 结果分析层(Result Analysis)                      │
│     └─ 内置的结果统计和报告                           │
│        ↓ 计算平均精度、标准差等                      │
│        └─ 生成CSV和TXT报告                          │
│                                                       │
└─────────────────────────────────────────────────────┘
```

---

## 两个集成脚本对比

| 特性 | ShipEar集成 | DeepShip集成 |
|------|-----------|-----------|
| 脚本文件 | `kfold_shipsear_integration.py` | `kfold_deepship_integration.py` |
| 训练方法 | ShipEar标准训练 | DeepShip蒸馏训练（学生网络+蒸馏） |
| 配置文件 | `configs/train_distillation_shipsear.yaml` | `configs/train_distillation_deepship.yaml` |
| 单Fold命令 | `--train-fold N` | `--fold N` |
| 全部Fold命令 | `--train-all` | `--all` |
| 输出目录 | `checkpoints/cv_shipsear/` | `results/kfold_cv_deepship/` |
| 结果CSV | `results/kfold_cv_shipsear/kfold_shipsear_results.csv` | `results/kfold_cv_deepship/kfold_results.csv` |
| 最适用场景 | 标准分类任务 | 需要模型蒸馏/压缩的场景 |

---

## ShipEar集成脚本使用

### 基本用法

#### 1. 生成K-Fold划分（首次使用）

```bash
python kfold_cross_validation.py
```

**输出**
```
results/kfold_splits/
├── kfold_summary.txt          # 汇总信息
├── kfold_fold00.txt ~ fold09.txt  # 各Fold详细
└── kfold_indices.txt          # 索引形式
```

#### 2. 验证生成结果

```bash
# 查看总体汇总
cat results/kfold_splits/kfold_summary.txt

# 查看Fold 0的样本数
grep "^[0-9]" results/kfold_splits/kfold_fold00.txt | wc -l
```

#### 3. 测试单个Fold

```bash
python kfold_shipsear_integration.py \
  --train-fold 0 \
  --gpus 4,5,6,7 \
  --epochs 100 \
  --batch-size 32
```

**参数说明**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--train-fold N` | 训练第N个Fold (0-9) | 无（必需） |
| `--gpus G1,G2,...` | 使用的GPU编号 | 4,5,6,7 |
| `--epochs E` | 训练轮数 | 100 |
| `--batch-size B` | 批大小 | 32 |
| `--learning-rate LR` | 学习率 | 0.001 |

**输出内容**
```
checkpoints/cv_shipsear/fold_00/
├── best_student.pth       # 最佳学生模型
├── latest.pth             # 最新检查点
├── config.yaml            # 训练配置
├── training.log           # 训练日志
└── metrics.csv            # 性能指标

results/kfold_cv_shipsear/
├── fold_00_metrics.txt    # Fold 0的性能汇总
└── kfold_shipsear_results.csv  # 追加结果
```

#### 4. 批量训练所有Fold

**选项A：顺序训练（推荐）**
```bash
python kfold_shipsear_integration.py \
  --train-all \
  --gpus 4,5,6,7 \
  --epochs 100
```

**优点**
- ✅ GPU显存清理完整
- ✅ 日志清晰易读
- ✅ 便于监控和调整
- ✅ 出错时容易恢复

**选项B：并行训练（高级）**
```bash
# 并行运行所有Fold
for fold_idx in {0..9}; do
    python kfold_shipsear_integration.py \
      --train-fold $fold_idx \
      --gpus 4,5,6,7 &
done

# 等待所有完成
wait
```

#### 5. 查看结果

```bash
# 显示格式化的最终结果
python kfold_shipsear_integration.py --results

# 查看详细CSV
cat results/kfold_cv_shipsear/kfold_shipsear_results.csv

# Python分析
python << 'EOF'
import pandas as pd
df = pd.read_csv('results/kfold_cv_shipsear/kfold_shipsear_results.csv')
print("平均准确率:", df['best_acc'].mean())
print("精度范围:", f"{df['best_acc'].min():.4f}-{df['best_acc'].max():.4f}")
print("标准差:", df['best_acc'].std())
EOF
```

### ShipEar完整参数列表

```bash
python kfold_shipsear_integration.py \
  --train-fold 0                              # 或 --train-all
  --gpus 4,5,6,7                              # GPU编号
  --data-dir ./data                           # 数据目录
  --base-config configs/train_distillation_shipsear.yaml  # 配置文件
  --splits-dir results/kfold_splits           # K-Fold划分目录
  --checkpoints-dir checkpoints/cv_shipsear   # 检查点目录
  --results-dir results/kfold_cv_shipsear     # 结果目录
```

---

## DeepShip集成脚本使用

### 基本用法

#### 1. 生成K-Fold划分（首次使用）

```bash
python kfold_cross_validation.py
```

#### 2. 测试单个Fold

```bash
python kfold_deepship_integration.py \
  --fold 0 \
  --config configs/train_distillation_deepship.yaml \
  --gpus 4,5,6,7
```

**参数说明**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--fold N` | 训练第N个Fold (0-9) | 无（需指定一个） |
| `--all` | 训练所有10个Fold | - |
| `--config FILE` | 训练配置文件 | `configs/train_distillation_deepship.yaml` |
| `--gpus G1,G2,...` | 使用的GPU编号 | 4,5,6,7 |
| `--splits-dir DIR` | K-Fold划分目录 | `results/kfold_splits` |
| `--results-dir DIR` | 结果保存目录 | `results/kfold_cv_deepship` |

#### 3. 批量训练所有Fold

```bash
python kfold_deepship_integration.py \
  --all \
  --config configs/train_distillation_deepship.yaml \
  --gpus 4,5,6,7
```

**输出内容**
```
results/kfold_cv_deepship/
├── fold_00/
│   ├── best_student.pth       # 最佳学生模型
│   ├── config.yaml            # 训练配置
│   └── training.log           # 训练日志
├── fold_01/ ~ fold_09/        # 其他Fold
├── kfold_results.csv          # 汇总结果表
└── kfold_summary_report.txt   # 总结报告
```

#### 4. 查看结果

```bash
# 查看CSV结果
cat results/kfold_cv_deepship/kfold_results.csv

# 查看总结报告
cat results/kfold_cv_deepship/kfold_summary_report.txt

# Python分析
python << 'EOF'
import pandas as pd
df = pd.read_csv('results/kfold_cv_deepship/kfold_results.csv')
print("平均验证精度:", df['best_val_acc'].mean())
print("精度范围:", f"{df['best_val_acc'].min():.4f}-{df['best_val_acc'].max():.4f}")
print("标准差:", df['best_val_acc'].std())
EOF
```

### DeepShip完整参数列表

```bash
python kfold_deepship_integration.py \
  --fold 0                                    # 或 --all
  --config configs/train_distillation_deepship.yaml  # 配置文件
  --gpus 4,5,6,7                              # GPU编号
  --splits-dir results/kfold_splits           # K-Fold划分目录
  --results-dir results/kfold_cv_deepship     # 结果保存目录
```

---

## 完整工作流程

### 【初次设置】第一次运行（15-30分钟）

```bash
# 第1步：运行交互式指南（可选但推荐）
python kfold_quick_start.py

# 或者手动操作
# 1. 编辑 kfold_cross_validation.py，修改 data_dir
# 2. 生成K-Fold划分
python kfold_cross_validation.py

# 3. 验证结果
cat results/kfold_splits/kfold_summary.txt

# 4. （推荐）提交到git
git add results/kfold_splits/
git commit -m "Add K-Fold splits"
```

### 【ShipEar训练流程】完整K-Fold训练（数小时到1天）

```bash
# 第1步：测试单个Fold（验证环境）
python kfold_shipsear_integration.py --train-fold 0 --gpus 4,5,6,7

# 第2步：查看Fold 0的结果
tail -f checkpoints/cv_shipsear/fold_00/training.log

# 第3步：开始完整训练（所有10个Fold）
python kfold_shipsear_integration.py --train-all --gpus 4,5,6,7

# 第4步：监控进度
watch -n 1 "ls checkpoints/cv_shipsear/fold_*/best_student.pth | wc -l"

# 第5步：查看最终结果
python kfold_shipsear_integration.py --results
```

### 【DeepShip训练流程】完整K-Fold蒸馏训练（数小时到1天）

```bash
# 第1步：测试单个Fold
python kfold_deepship_integration.py --fold 0 --gpus 4,5,6,7

# 第2步：查看训练日志
tail -f results/kfold_cv_deepship/fold_00/training.log

# 第3步：开始完整训练
python kfold_deepship_integration.py --all --gpus 4,5,6,7

# 第4步：监控进度
watch -n 1 "ls results/kfold_cv_deepship/fold_*/best_student.pth 2>/dev/null | wc -l"

# 第5步：查看结果摘要
cat results/kfold_cv_deepship/kfold_summary_report.txt

# 第6步：查看详细CSV
cat results/kfold_cv_deepship/kfold_results.csv
```

### 【结果比较】对比两个方法的结果

```bash
# 使用Python进行对比分析
python << 'EOF'
import pandas as pd

print("=" * 80)
print("ShipEar vs DeepShip K-Fold 交叉验证结果对比")
print("=" * 80 + "\n")

# 读取ShipEar结果
df_shipsear = pd.read_csv('results/kfold_cv_shipsear/kfold_shipsear_results.csv')
print("【ShipEar 结果】")
print(f"平均准确率: {df_shipsear['best_acc'].mean():.4f} ± {df_shipsear['best_acc'].std():.4f}")
print(f"准确率范围: {df_shipsear['best_acc'].min():.4f} - {df_shipsear['best_acc'].max():.4f}\n")

# 读取DeepShip结果
df_deepship = pd.read_csv('results/kfold_cv_deepship/kfold_results.csv')
print("【DeepShip 结果】")
print(f"平均验证精度: {df_deepship['best_val_acc'].mean():.4f} ± {df_deepship['best_val_acc'].std():.4f}")
print(f"精度范围: {df_deepship['best_val_acc'].min():.4f} - {df_deepship['best_val_acc'].max():.4f}\n")

print("【性能对比】")
diff = df_deepship['best_val_acc'].mean() - df_shipsear['best_acc'].mean()
print(f"性能差异: {diff:+.4f}")
print(f"改进方向: {'✓ DeepShip更优' if diff > 0 else '✓ ShipEar更优'}")
EOF
```

---

## 命令参考

### K-Fold划分生成

```bash
# 生成K-Fold划分（包含默认参数）
python kfold_cross_validation.py

# 自定义参数生成
python kfold_cross_validation.py \
  --data-dir /path/to/data \
  --num-folds 10 \
  --seed 42

# 查看总体汇总
cat results/kfold_splits/kfold_summary.txt

# 统计样本数
for i in {0..9}; do
  echo "Fold $i: $(grep '^[0-9]' results/kfold_splits/kfold_fold0$i.txt | wc -l) 样本"
done
```

### ShipEar单个Fold训练

```bash
# 基本训练
python kfold_shipsear_integration.py --train-fold 0 --gpus 4,5,6,7

# 自定义超参数
python kfold_shipsear_integration.py \
  --train-fold 0 \
  --gpus 4,5,6,7 \
  --epochs 150 \
  --batch-size 64 \
  --learning-rate 0.0005

# 恢复中断的训练
python kfold_shipsear_integration.py \
  --train-fold 0 \
  --gpus 4,5,6,7 \
  --resume-from checkpoints/cv_shipsear/fold_00/latest.pth
```

### ShipEar批量训练

```bash
# 训练所有Fold（顺序）
python kfold_shipsear_integration.py --train-all --gpus 4,5,6,7

# 查看ShipEar结果
python kfold_shipsear_integration.py --results

# 显示详细CSV
cat results/kfold_cv_shipsear/kfold_shipsear_results.csv

# 监控GPU使用
watch -n 1 nvidia-smi

# 查看训练日志
tail -f checkpoints/cv_shipsear/fold_00/training.log
```

### DeepShip单个Fold训练

```bash
# 基本训练
python kfold_deepship_integration.py --fold 0 --gpus 4,5,6,7

# 自定义配置
python kfold_deepship_integration.py \
  --fold 0 \
  --config configs/train_distillation_deepship.yaml \
  --gpus 4,5,6,7
```

### DeepShip批量训练

```bash
# 训练所有Fold
python kfold_deepship_integration.py --all --gpus 4,5,6,7

# 查看DeepShip结果
cat results/kfold_cv_deepship/kfold_results.csv

# 查看总结报告
cat results/kfold_cv_deepship/kfold_summary_report.txt

# 监控训练进度
watch -n 5 "tail results/kfold_cv_deepship/fold_00/training.log"
```

### 工具和诊断

```bash
# 验证K-Fold划分
python kfold_cross_validation.py --validate

# 检查每个Fold的类别分布
python << 'EOF'
import numpy as np
for i in range(10):
    data = np.loadtxt(f'results/kfold_splits/kfold_fold{i:02d}.txt', dtype=str)
    classes = data[:, 1]
    unique, counts = np.unique(classes, return_counts=True)
    print(f"Fold {i:02d}:", dict(zip(unique, counts)))
EOF

# 检查训练进度
ls -lt checkpoints/cv_shipsear/*/latest.pth | head -5

# 查看最新的训练日志
tail -50 $(ls -t checkpoints/cv_shipsear/fold_*/training.log | head -1)
```

---

## 常见问题

### Q1: 应该使用ShipEar还是DeepShip？

**回答**：
- **ShipEar** - 用于标准分类任务，模型训练和评估
- **DeepShip** - 用于需要模型蒸馏的场景（如模型压缩、知识转移）

选择标准：看你的任务需要什么。两者都支持K-Fold交叉验证，可以对比结果。

### Q2: K-Fold划分数据后不平衡怎么办？

**问题描述**: 某些Fold的类别分布不均衡。

**解决方案**:
K-Fold生成脚本默认使用 `StratifiedKFold`，确保每个Fold都有相同的类别分布比例。
如果仍有问题，检查原始数据是否平衡：

```python
# 查看原始数据类别分布
python << 'EOF'
import numpy as np
import os

# 检查每个Fold的类别分布
for i in range(10):
    fold_file = f'results/kfold_splits/kfold_fold{i:02d}.txt'
    data = np.loadtxt(fold_file, dtype=str, skiprows=1)
    classes = data[:, 1]
    unique, counts = np.unique(classes, return_counts=True)
    print(f"Fold {i}: {dict(zip(unique, counts))}")
EOF
```

### Q3: 如何恢复中断的训练？

**问题描述**: 训练过程中发生中断（如掉电、手动停止等）。

**解决方案**:
- **ShipEar**:
```bash
# 检查latest.pth是否存在
ls checkpoints/cv_shipsear/fold_00/latest.pth

# 恢复训练
python kfold_shipsear_integration.py \
  --train-fold 0 \
  --gpus 4,5,6,7 \
  --resume-from checkpoints/cv_shipsear/fold_00/latest.pth
```

- **DeepShip**:
```bash
# DeepShip会自动从上一次的检查点恢复
python kfold_deepship_integration.py --fold 0 --gpus 4,5,6,7
```

### Q4: GPU显存不足？

**问题描述**: CUDA out of memory 错误。

**解决方案**:

- **方法1**：减小批大小（ShipEar）
```bash
python kfold_shipsear_integration.py \
  --train-fold 0 \
  --gpus 4,5,6,7 \
  --batch-size 16  # 从32改为16
```

- **方法2**：使用单个GPU
```bash
python kfold_shipsear_integration.py \
  --train-fold 0 \
  --gpus 4 \
  --batch-size 16
```

- **方法3**：使用CPU（但很慢）
```bash
python kfold_shipsear_integration.py \
  --train-fold 0 \
  --device cpu \
  --batch-size 8
```

### Q5: 如何在代码中使用K-Fold数据？

**问题描述**: 想在自己的训练脚本中使用生成的K-Fold数据。

**解决方案**:
```python
from kfold_data_loader import KFoldDataLoader

# 初始化加载器
loader = KFoldDataLoader(
    splits_dir='results/kfold_splits',
    fold_idx=0
)

# 获取训练和验证数据
train_samples = loader.get_train_samples()
val_samples = loader.get_val_samples()

# 使用数据进行训练
for sample in train_samples:
    # 处理样本
    pass
```

### Q6: 如何比较不同的超参数配置？

**问题描述**: 想用K-Fold进行超参数调优。

**解决方案**:
```bash
# 创建多个配置文件
mkdir configs/exp_configs
cp configs/train_distillation_shipsear.yaml configs/exp_configs/config_v1.yaml
cp configs/train_distillation_shipsear.yaml configs/exp_configs/config_v2.yaml

# 用不同配置训练
for config in configs/exp_configs/*.yaml; do
    python kfold_shipsear_integration.py \
      --train-all \
      --config $config \
      --gpus 4,5,6,7
done

# 对比结果
python << 'EOF'
import pandas as pd
import os

results = {}
for config_name in ['v1', 'v2']:
    results_file = f'results/exp_configs/{config_name}/kfold_shipsear_results.csv'
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        results[config_name] = {
            'mean': df['best_acc'].mean(),
            'std': df['best_acc'].std()
        }

for name, stats in results.items():
    print(f"{name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
EOF
```

---

## 故障排查

### 错误：FileNotFoundError: results/kfold_splits/...

**原因**: K-Fold划分还未生成。

**解决**:
```bash
# 先生成划分
python kfold_cross_validation.py

# 验证文件存在
ls results/kfold_splits/

# 验证文件内容
head -10 results/kfold_splits/kfold_fold00.txt
```

### 错误：CUDA out of memory

**原因**: GPU显存不足。

**解决**:
```bash
# 方法1：减小批大小
--batch-size 16

# 方法2：使用单GPU
--gpus 4

# 方法3：梯度累积
--gradient-accumulation-steps 2

# 方法4：混合精度训练
--fp16

# 使用示例
python kfold_shipsear_integration.py \
  --train-fold 0 \
  --gpus 4 \
  --batch-size 16
```

### 错误：ModuleNotFoundError

**原因**: 训练脚本找不到模块。

**解决**:
```bash
# 确保在项目根目录运行
cd /path/to/project

# 检查Python路径
export PYTHONPATH=/path/to/project:$PYTHONPATH

# 验证导入
python -c "from kfold_data_loader import KFoldDataLoader"
```

### 错误：Config file not found

**原因**: 配置文件路径错误。

**解决**:
```bash
# 检查配置文件是否存在
ls -la configs/train_distillation_*.yaml

# 使用正确的路径运行
python kfold_shipsear_integration.py \
  --train-fold 0 \
  --base-config configs/train_distillation_shipsear.yaml \
  --gpus 4,5,6,7
```

### 错误：RuntimeError: Expected all tensors to be on the same device

**原因**: 数据和模型不在同一设备上。

**解决**:
```python
# 确保在加载数据时转移到GPU
import torch

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

# 数据处理
data = data.to(device)

# 模型
model = model.to(device)
```

### 问题：结果不一致

**原因**: 随机种子未固定或并行训练。

**解决**:
```python
# 设置随机种子
import torch
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 如果使用多GPU并行，禁用随机性
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 确保单GPU训练
export CUDA_VISIBLE_DEVICES=4
```

### 问题：训练速度慢

**原因**: 可能是数据加载、GPU利用率低、或批大小太小。

**诊断**:
```bash
# 监控GPU使用
watch -n 1 nvidia-smi

# 查看数据加载时间
# 在日志中查找 "Data loading time"

# 解决方案
# 1. 增加批大小（如果显存允许）
--batch-size 64

# 2. 增加数据加载线程
--num-workers 8

# 3. 使用多GPU并行
--gpus 4,5,6,7
```

---

## 文件结构总览

### 核心文件
```
├── kfold_cross_validation.py           # 生成K-Fold划分
├── kfold_data_loader.py                # 加载K-Fold数据（Python API）
├── kfold_shipsear_integration.py       # ShipEar集成脚本
├── kfold_deepship_integration.py       # DeepShip集成脚本
└── K-FOLD_TRAINING_INTEGRATION_GUIDE.md # 本文档
```

### 生成的结果
```
results/
├── kfold_splits/                       # K-Fold划分
│   ├── kfold_summary.txt
│   ├── kfold_fold00.txt ~ fold09.txt
│   └── kfold_indices.txt
│
├── kfold_cv_shipsear/                  # ShipEar训练结果
│   ├── kfold_shipsear_results.csv
│   ├── training_summary.txt
│   └── fold_00/ ~ fold_09/
│
└── kfold_cv_deepship/                  # DeepShip训练结果
    ├── kfold_results.csv
    ├── kfold_summary_report.txt
    └── fold_00/ ~ fold_09/

checkpoints/cv_shipsear/                # ShipEar模型检查点
├── fold_00/
│   ├── best_student.pth
│   ├── config.yaml
│   └── training.log
└── fold_01/ ~ fold_09/
```

---

## 最佳实践

### 1. 数据管理
- ✅ 用 `git` 追踪 `results/kfold_splits/`（保证可复现）
- ✅ 定期备份 `results/kfold_cv_*/`（防止丢失）
- ✅ 定期备份 `checkpoints/`（防止丢失）
- ❌ 不要手动修改生成的txt文件

### 2. 训练监控
- ✅ 使用 `tail -f` 实时监控日志
- ✅ 定期检查GPU使用 (`nvidia-smi`)
- ✅ 记录不同配置的性能
- ✅ 预估总训练时间（Fold数 × 单Fold时间）

### 3. 结果管理
- ✅ 为不同实验创建不同目录
- ✅ 保存配置文件和结果一起
- ✅ 记录运行的完整命令
- ✅ 提交重要结果到版本控制

### 4. 性能优化
- ✅ 使用多GPU训练加速
- ✅ 调整批大小平衡速度和显存
- ✅ 使用混合精度训练（fp16）降低显存
- ✅ 启用梯度累积进行大批大小训练

### 5. 可复现性
- ✅ 固定随机种子
- ✅ 记录运行环境（Python版本、库版本、GPU型号）
- ✅ 保存所有配置和超参数
- ✅ 提交训练日志和结果

---

## 使用工作流总结

### 推荐流程

```
Step 1: 生成K-Fold划分
↓
python kfold_cross_validation.py

Step 2: 选择训练方法
├─→ ShipEar标准训练
│   ├─ python kfold_shipsear_integration.py --train-fold 0 （测试）
│   └─ python kfold_shipsear_integration.py --train-all （完整）
│
└─→ DeepShip蒸馏训练
    ├─ python kfold_deepship_integration.py --fold 0 （测试）
    └─ python kfold_deepship_integration.py --all （完整）

Step 3: 查看结果
├─ ShipEar: python kfold_shipsear_integration.py --results
└─ DeepShip: cat results/kfold_cv_deepship/kfold_summary_report.txt

Step 4: 对比和分析
└─ 使用Python pandas分析和对比结果
```

---

## 总结

这个K-Fold十折交叉验证系统提供了：

✅ **两种训练方法** - ShipEar标准训练和DeepShip蒸馏训练  
✅ **自动化的数据划分** - 确保平衡和可复现  
✅ **灵活的训练执行** - 支持单个或批量Fold  
✅ **详细的结果分析** - 完整的性能指标统计  
✅ **易用的API** - 简化代码集成  

**下一步**：
1. 根据你的任务选择合适的训练方法
2. 按照本指南的流程进行训练
3. 对比两种方法的结果，选择更优的方案

祝你训练顺利！🚀

---

**文档版本**: 1.0  
**更新时间**: 2025-05-06  
**适用平台**: Linux/macOS/Windows  
**Python版本**: 3.7+  
**主要依赖**: torch, scikit-learn, numpy, pyyaml
