# batch_eval_acc_auc_f1_shipsear.py 使用指南

本脚本用于评估在 ShipsEar 数据集上训练的学生模型，计算 Accuracy、AUC 和 F1 Score 等关键指标。

## 功能特性

- **支持的数据集**: ShipsEar（4 类船舶声学信号分类）
- **评估指标**:
  - Accuracy（准确率）
  - AUC（曲线下面积）：支持多分类（OVO 和 OVR）
  - F1 Score：提供 Macro、Micro 和 Weighted 三种平均方式
- **支持的评估子集**: train / validation / test
- **输出内容**:
  - 终端显示：所有指标、混淆矩阵、分类报告
  - 文件保存：`evaluation_results_{dataset}.txt`

## 基本用法

### 1. 最简单的使用方式（使用默认配置）

```bash
python evaluate_student_shipsear.py \
  --checkpoint /path/to/best_student.pth
```

这将在测试集上评估模型，使用默认配置文件 `configs/train_distillation_shipsear.yaml`。

### 2. 指定评估的数据子集

```bash
# 在测试集上评估
python evaluate_student_shipsear.py \
  --checkpoint /path/to/best_student.pth \
  --dataset test

# 在验证集上评估
python evaluate_student_shipsear.py \
  --checkpoint /path/to/best_student.pth \
  --dataset validation

# 在训练集上评估
python evaluate_student_shipsear.py \
  --checkpoint /path/to/best_student.pth \
  --dataset train
```

### 3. 指定 GPU 和批处理大小

```bash
python evaluate_student_shipsear.py \
  --checkpoint /path/to/best_student.pth \
  --gpu 0 \
  --batch_size 64
```

### 4. 使用自定义配置文件

```bash
python evaluate_student_shipsear.py \
  --checkpoint /path/to/best_student.pth \
  --config /path/to/your_config.yaml
```

## 参数说明

| 参数 | 类型 | 默认值 | 必需 | 说明 |
|------|------|--------|------|------|
| `--checkpoint` | str | - | ✅ | 模型权重文件路径（`best_student.pth`） |
| `--config` | str | `configs/train_distillation_shipsear.yaml` | ❌ | 配置文件路径 |
| `--batch_size` | int | 32 | ❌ | 批处理大小 |
| `--gpu` | str | 0 | ❌ | 使用的 GPU ID |
| `--dataset` | str | test | ❌ | 评估的数据子集（train/validation/test） |

## 输出示例

### 终端输出

```
Using device: cuda
Config loaded from: /media/hdd1/fubohan/Code/UATR/configs/train_distillation_shipsear.yaml

Dataset:
  Data dir: /path/to/ShipsEar
  Data type: wav_s@wav_t
  Dataset to evaluate: test

Model:
  Num classes: 4
  p_encoder: 0.2
  p_classifier: 0.3
  Checkpoint: /path/to/best_student.pth

Loading test dataset...
  Dataset size: 312

Creating student model...
Loading checkpoint from: /path/to/best_student.pth
  Best accuracy during training: 92.15%
  Checkpoint loaded successfully!

Evaluating model on test set...

============================================================
EVALUATION RESULTS
============================================================

Accuracy: 92.3077%

F1 Score:
  Macro:    91.5385%
  Micro:    92.3077%
  Weighted:  92.3105%

AUC Score:
  OVO (One-vs-One):   98.1234%
  OVR (One-vs-Rest):  98.2345%

Confusion Matrix:
[[78  1  2  0]
 [ 1 73  3  1]
 [ 3  2 70  2]
 [ 0  1  2 74]]

Classification Report:
                precision    recall  f1-score   support

passengership     0.9512    0.9630    0.9571        81
tanker            0.9516    0.9342    0.9428        78
cargo             0.9091    0.9032    0.9061        77
tug               0.9487    0.9462    0.9475        79

accuracy                                0.9369       315
macro avg         0.9402    0.9367    0.9384       315
weighted avg      0.9402    0.9369    0.9385       315

Per-class Metrics:
Class           Precision    Recall       F1-Score     Support
-----------------------------------------------------------------
passengership   0.9512       0.9630       0.9571       81
tanker          0.9516       0.9342       0.9428       78
cargo           0.9091       0.9032       0.9061       77
tug             0.9487       0.9462       0.9475       79

============================================================
EVALUATION COMPLETED
============================================================

Results saved to: /path/to/evaluation_results_test.txt
```

### 保存的文件内容

评估结果会自动保存为 `evaluation_results_{dataset}.txt` 文件，包含以下内容：

```
============================================================
EVALUATION RESULTS - TEST SET
============================================================

Checkpoint: /path/to/best_student.pth
Dataset: /path/to/ShipsEar
Dataset split: test
Number of samples: 312

Accuracy: 92.3077%

F1 Score:
  Macro:    91.5385%
  Micro:    92.3077%
  Weighted:  92.3105%

AUC Score:
  OVO (One-vs-One):   98.1234%
  OVR (One-vs-Rest):  98.2345%

Confusion Matrix:
[[78  1  2  0]
 [ 1 73  3  1]
 [ 3  2 70  2]
 [ 0  1  2 74]]

Classification Report:
                precision    recall  f1-score   support

passengership     0.9512    0.9630    0.9571        81
tanker            0.9516    0.9342    0.9428        78
cargo             0.9091    0.9032    0.9061        77
tug               0.9487    0.9462    0.9475        79

accuracy                                0.9369       315
macro avg         0.9402    0.9367    0.9384       315
weighted avg      0.9402    0.9369    0.9385       315
```

## 实际应用示例

### 示例 1：评估单个模型

```bash
python evaluate_student_shipsear.py \
  --checkpoint /media/hdd1/fubohan/Code/UATR/checkpoints/ShipsEar/comparison_distillation_kd/best_student.pth \
  --gpu 0
```

### 示例 2：评估所有蒸馏方法

创建一个批量评估脚本 `batch_evaluate.sh`：

```bash
#!/bin/bash

# 定义蒸馏方法列表
METHODS=("kd" "fitnet" "at" "fsp" "pkd" "pkt" "rkd" "ab" "srgd" "ccd" "crd" \
         "cc" "nst" "similarity" "vid" "sp" "mgd" "mkd" "nkd" "ickd" "lskd" "cat_kd" "sdd" "vkd")

# 批量评估
for method in "${METHODS[@]}"; do
    echo "Evaluating $method..."
    CHECKPOINT="/media/hdd1/fubohan/Code/UATR/checkpoints/ShipsEar/comparison_distillation_${method}/best_student.pth"
    
    if [ -f "$CHECKPOINT" ]; then
        python evaluate_student_shipsear.py \
            --checkpoint "$CHECKPOINT" \
            --dataset test \
            --gpu 0
        echo "----------------------------------------"
    else
        echo "Checkpoint not found: $CHECKPOINT"
    fi
done
```

### 示例 3：评估不同数据子集

```bash
# 评估所有子集
python evaluate_student_shipsear.py \
  --checkpoint /path/to/best_student.pth \
  --dataset train \
  --gpu 0

python evaluate_student_shipsear.py \
  --checkpoint /path/to/best_student.pth \
  --dataset validation \
  --gpu 0

python evaluate_student_shipsear.py \
  --checkpoint /path/to/best_student.pth \
  --dataset test \
  --gpu 0
```

## 常见问题

### Q1: 找不到模型权重文件

**问题**: 提示 `FileNotFoundError` 或 `checkpoint not found`

**解决方法**:
- 检查 `--checkpoint` 参数指定的路径是否正确
- 确保训练已完成并生成了 `best_student.pth` 文件
- 使用绝对路径避免路径问题

### Q2: CUDA out of memory

**问题**: 提示显存不足

**解决方法**:
- 减小 `--batch_size` 参数，例如设置为 16 或 8
- 使用不同的 GPU：`--gpu 1`

### Q3: 数据集加载失败

**问题**: 提示数据集路径错误

**解决方法**:
- 检查配置文件中的 `dataset.data_dir` 是否正确
- 确保数据集已经正确放置在指定目录
- 数据集结构应符合 ShipsEar 数据集的标准格式

### Q4: 配置文件路径错误

**问题**: 提示配置文件不存在

**解决方法**:
- 使用 `--config` 参数指定正确的配置文件路径
- 确保 YAML 文件格式正确
- 检查 `configs/train_distillation_shipsear.yaml` 是否存在

## ShipsEar 数据集类别

脚本会自动识别 4 个类别：
- `passengership`（客船）
- `tanker`（油轮）
- `cargo`（货船）
- `tug`（拖船）

## 依赖要求

确保已安装以下依赖：
```bash
pip install torch torchvision numpy scikit-learn pyyaml
```

## 相关文件

- 训练脚本: `run_20_distillation_methods_shipsear.py`
- 配置文件: `configs/train_distillation_shipsear.yaml`
- 数据集类: `datasets/audio_dataset.py`
- 模型定义: `models/LNN.py`

## 注意事项

1. **GPU 使用**: 脚本默认使用 GPU 0，如需使用其他 GPU，请通过 `--gpu` 参数指定
2. **批处理大小**: 根据显存大小调整 `--batch_size`，默认为 32
3. **结果保存**: 评估结果会自动保存在模型权重的同一目录下
4. **性能监控**: 评估过程中会显示当前加载的批次进度
5. **多模态输入**: 脚本会自动处理 `wav_s@wav_t` 格式的输入，学生模型只使用第一个模态

## 技术支持

如遇到问题，请检查：
1. 模型权重文件路径是否正确
2. 数据集路径是否正确
3. 配置文件格式是否正确
4. GPU 是否可用
5. 所有依赖包是否已正确安装
