# K-Fold 50折叠交叉验证使用指南

## 概述

此文档详细介绍如何使用K-Fold 50折叠交叉验证系统对DeepShip和ShipsEar数据进行模型训练和评估。

**主要特性：**
- 50折叠分层交叉验证
- 数据自动划分和保存
- 支持多个数据集（DeepShip、ShipsEar）
- 集成蒸馏模型训练
- 自动结果收集和报告生成

---

## 核心组件

### 1. **kfold_cross_validation.py**
主程序：生成K-Fold数据划分
- **功能**：扫描数据目录，生成平衡的50个折叠，保存划分结果
- **输出**：
  - `kfold_fold00.txt` - `kfold_fold49.txt`：每个折叠的详细数据列表
  - `kfold_summary.txt`：所有折叠的统计汇总
  - `kfold_indices.txt`：索引形式的划分结果

### 2. **kfold_data_loader.py**
数据加载器：在训练脚本中方便地加载划分数据
- **功能**：读取K-Fold文件，解析样本路径
- **使用示例**：
```python
from kfold_data_loader import KFoldDataLoader

loader = KFoldDataLoader("results/kfold_splits/", fold_idx=0)
train_samples = loader.get_train_samples()
val_samples = loader.get_val_samples()
```

### 3. **kfold_deepship_integration.py**
DeepShip模型 + K-Fold集成脚本
- **功能**：为每个Fold运行蒸馏训练，收集结果

### 4. **kfold_shipsear_integration.py**
ShipsEar模型 + K-Fold集成脚本
- **功能**：为每个Fold运行蒸馏训练，收集结果

---

## 快速开始

### 步骤1：生成K-Fold划分

#### 生成ShipsEar数据集的50折叠划分
```bash
python experiments/cv/kfold_cross_validation.py \
  --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622 \
  --output-dir results/kfold_splits_shipsear
```

#### 生成DeepShip数据集的50折叠划分
```bash
python experiments/cv/kfold_cross_validation.py \
  --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/DeepShip_622 \
  --output-dir results/kfold_splits_deepship
```

**参数说明：**
- `--data-dir`：数据集所在目录（必须包含train/wav/[类别]/*.wav结构）
- `--output-dir`：划分结果保存目录（默认：results/kfold_splits）

**输出内容：**
```
results/kfold_splits_shipsear/
├── kfold_fold00.txt       # Fold 0 的详细样本列表
├── kfold_fold01.txt       # Fold 1 的详细样本列表
├── ...
├── kfold_fold49.txt       # Fold 49 的详细样本列表
├── kfold_summary.txt      # 所有折叠的统计汇总
└── kfold_indices.txt      # 索引形式（便于Python加载）
```

---

### 步骤2：训练模型

#### 使用DeepShip模型

**训练单个Fold（Fold 0）：**
```bash
python kfold_deepship_integration.py \
  --fold 0 \
  --splits-dir results/kfold_splits_deepship \
  --results-dir results/kfold_cv_deepship \
  --config configs/train_distillation_deepship.yaml \
  --gpus 4,5,6,7
```

**训练所有50个Fold：**
```bash
python kfold_deepship_integration.py \
  --all \
  --splits-dir results/kfold_splits_deepship \
  --results-dir results/kfold_cv_deepship \
  --config configs/train_distillation_deepship.yaml \
  --gpus 4,5,6,7
```

#### 使用ShipsEar模型

**第一步：生成K-Fold划分**
```bash
python kfold_shipsear_integration.py \
  --setup \
  --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622
```

**第二步：训练所有50个Fold**
```bash
python kfold_shipsear_integration.py \
  --train-all \
  --gpus 4,5,6,7 \
  --splits-dir results/kfold_splits_shipsear \
  --results-dir results/kfold_cv_shipsear
```

**第三步：查看结果**
```bash
python kfold_shipsear_integration.py --results
```

**或只训练单个Fold（用于测试）：**
```bash
python kfold_shipsear_integration.py \
  --train-fold 0 \
  --gpus 4,5,6,7
```

**参数说明：**
- `--fold N` 或 `--train-fold N`：训练指定的Fold（0-49）
- `--all` 或 `--train-all`：训练所有50个Fold
- `--gpus`：GPU编号列表（默认：4,5,6,7）
- `--splits-dir`：K-Fold划分文件目录
- `--results-dir`：训练结果保存目录
- `--config`：训练配置文件路径

---

## 完整工作流示例

### 场景：在多个终端上并行训练

**终端1：准备数据**
```bash
# 为DeepShip生成K-Fold划分
python experiments/cv/kfold_cross_validation.py \
  --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/DeepShip_622 \
  --output-dir results/kfold_splits_deepship

# 为ShipsEar生成K-Fold划分
python experiments/cv/kfold_cross_validation.py \
  --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622 \
  --output-dir results/kfold_splits_shipsear
```

**终端2：训练DeepShip（所有50个Fold）**
```bash
cd /media/hdd1/fubohan/Code/UATR
CUDA_VISIBLE_DEVICES=4,5,6,7 python kfold_deepship_integration.py \
  --all \
  --gpus 0,1,2,3 \
  --splits-dir results/kfold_splits_deepship \
  --results-dir results/kfold_cv_deepship
```

**终端3：训练ShipsEar（所有50个Fold）**
```bash
cd /media/hdd1/fubohan/Code/UATR
CUDA_VISIBLE_DEVICES=4,5,6,7 python kfold_shipsear_integration.py \
  --train-all \
  --gpus 0,1,2,3 \
  --splits-dir results/kfold_splits_shipsear \
  --results-dir results/kfold_cv_shipsear
```

2026/5/11运行版本

Deephip数据集
```
# 训练所有50个Fold（分5组，每组10折叠）
python kfold_deepship_integration.py --all --gpus 4,5,6,7

# 或只训练某个Fold进行测试
python kfold_deepship_integration.py --fold 0 --gpus 4,5,6,7

# 查看结果
cat results/kfold_cv_deepship/kfold_summary_Group_0.txt
cat results/kfold_cv_deepship/kfold_summary_report.txt  # 总体报告
```

shipsear版本
```
# 第1步：生成K-Fold划分（如果还没有）
python kfold_shipsear_integration.py --setup --data-dir ./data

# 第2步：训练所有50个Fold（分5组，每组10折叠）
python kfold_shipsear_integration.py --train-all --gpus 4,5,6,7

# 或只训练某个Fold进行测试
python kfold_shipsear_integration.py --train-fold 0 --gpus 4,5,6,7

# 查看结果
python kfold_shipsear_integration.py --results

# 指定保存目录
python kfold_shipsear_integration.py --all --results-dir results/kfold_cv_shipsear --gpus 4,5,6,7
```

---

## 输出结果说明

### 训练结果目录结构

```
results/kfold_cv_deepship/
├── fold_00/
│   ├── config.yaml                    # Fold 0 的配置文件
│   ├── best_student.pth               # 最佳学生模型
│   ├── training.log                   # 训练日志
│   └── ...
├── fold_01/
├── ...
├── fold_49/
├── kfold_results.csv                  # 汇总结果（CSV格式）
└── kfold_summary_report.txt           # 总结报告
```

### kfold_results.csv 格式

| fold_idx | n_train | n_val | best_train_acc | best_val_acc | best_epoch | training_time | status  |
|----------|---------|-------|----------------|--------------|------------|---------------|---------|
| 0        | 2800    | 560   | 0.95           | 0.92         | 180        | 3600.5        | success |
| 1        | 2800    | 560   | 0.94           | 0.91         | 175        | 3550.3        | success |
| ...      | ...     | ...   | ...            | ...          | ...        | ...           | ...     |

### kfold_summary_report.txt 内容示例

```
================================================================================
DeepShip K-Fold 50折叠交叉验证 - 训练总结报告
生成时间: 2026-05-07 10:30:45
================================================================================

【验证集精度统计】
平均精度:  0.8950
最高精度:  0.9200
最低精度:  0.8600
标准差:    0.0180

【各Fold精度】
Fold 0: train_acc=0.9500, val_acc=0.9200, epoch=180
Fold 1: train_acc=0.9400, val_acc=0.9100, epoch=175
...
Fold 49: train_acc=0.9350, val_acc=0.9050, epoch=185

================================================================================
详细结果CSV: kfold_results.csv
================================================================================
```

---

## 常见问题

### Q1: 如何只训练某个Fold进行测试？
**A:** 使用 `--fold` 或 `--train-fold` 参数指定单个Fold
```bash
python kfold_deepship_integration.py --fold 0 --gpus 4,5,6,7
```

### Q2: K-Fold划分文件在哪里？
**A:** 在 `--output-dir` 指定的目录下，默认为 `results/kfold_splits/`
```
results/kfold_splits/
├── kfold_fold00.txt
├── kfold_fold01.txt
├── ...
├── kfold_fold49.txt
├── kfold_summary.txt
└── kfold_indices.txt
```

### Q3: 如何查看训练进度？
**A:** 查看实时训练日志
```bash
tail -f results/kfold_cv_deepship/fold_00/training.log
```

### Q4: 如何重新生成某个Fold的数据？
**A:** 删除该Fold的文件，重新运行划分脚本
```bash
rm results/kfold_splits/kfold_fold00.txt
python experiments/cv/kfold_cross_validation.py --data-dir ...
```

### Q5: 不同数据集是否可以使用相同的Fold划分？
**A:** 不建议。虽然技术上可行，但不同数据集的类别分布可能不同，应该为每个数据集独立生成Fold划分。

### Q6: 如何使用已有的K-Fold划分文件？
**A:** 在训练脚本中指定 `--splits-dir` 参数
```bash
python kfold_deepship_integration.py \
  --all \
  --splits-dir results/kfold_splits_deepship  # 指定已有的划分文件目录
```

---

## 性能建议

### 计算时间估计

- **K-Fold划分生成**：取决于数据集大小，通常 < 5 分钟
- **单Fold训练时间**：取决于模型大小和GPU数量
  - DeepShip（每Fold）：~30-60 分钟（4个GPU）
  - ShipsEar（每Fold）：~20-40 分钟（4个GPU）
- **全部50Fold训练时间**：
  - 单GPU顺序执行：~50-100 小时
  - 4个GPU（4块GPU）：~12-25 小时

### GPU内存要求

- **推荐配置**：每个模型4块GPU
- **最小配置**：2块GPU（会降低速度）
- **内存需求**：每块GPU需要 ≥ 12GB



---

## 数据集准备

### 数据目录结构要求

```
ShipsEar_622/
├── train/
│   └── wav/
│       ├── class_0/
│       │   ├── sample_001.wav
│       │   ├── sample_002.wav
│       │   └── ...
│       ├── class_1/
│       │   ├── sample_001.wav
│       │   └── ...
│       └── ...
├── validation/
│   └── wav/
│       └── ... (可选)
└── test/
    └── wav/
        └── ... (可选)
```

**要求：**
- 每个类别在单独的子目录中
- 音频文件格式：WAV
- 训练集必须存在，验证集和测试集可选

---

## 参考

- **主论文**：K-Fold Cross-Validation for Model Evaluation
- **相关文件**：
  - `experiments/cv/kfold_cross_validation.py`：主程序
  - `kfold_data_loader.py`：数据加载器
  - `kfold_deepship_integration.py`：DeepShip集成
  - `kfold_shipsear_integration.py`：ShipsEar集成

---

**最后更新：2026-05-11**
