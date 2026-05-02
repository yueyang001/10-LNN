# DeepShip 模型评估系统 - 设置总结

## 概述

已为你创建了一套完整的评估系统,用于计算 DeepShip 数据集上训练的学生模型的 ACC、AUC、F1 等指标。

## 创建的文件

### 1. 核心评估脚本

#### `evaluate_student_deepship.py`
- **功能**: 评估单个 `best_student.pth` 模型
- **输出指标**:
  - Accuracy (准确率)
  - F1-Score (Macro/Micro/Weighted)
  - AUC (One-vs-One / One-vs-Rest)
  - Confusion Matrix (混淆矩阵)
  - Per-class Precision/Recall/F1-Score
- **输出文件**: 自动保存在 checkpoint 同目录下,命名为 `evaluation_results_{dataset}.txt`

#### `batch_evaluate_ablation_deepship.py`
- **功能**: 批量评估 `checkpoints/ablation_deepship/` 下所有模型
- **特性**:
  - 自动发现所有包含 `best_student.pth` 的目录
  - 逐个评估所有模型
  - 生成按 Accuracy 排名的汇总报告
  - 统计信息(平均准确率、标准差等)
- **输出文件**: `evaluation_summary_{dataset}.txt`

### 2. 便捷启动脚本

#### `evaluate_single.sh`
- **用途**: 快速评估单个模型
- **使用方法**:
  ```bash
  ./evaluate_single.sh <checkpoint_path> [gpu_id] [dataset]
  ```

#### `evaluate_all_ablation.sh`
- **用途**: 一键评估所有 ablation 模型
- **使用方法**:
  ```bash
  ./evaluate_all_ablation.sh
  ```

### 3. 文档

#### `EVALUATION_README.md`
- 详细的使用说明
- 参数说明
- 指标解释
- 故障排除指南

## 快速开始

### 方式 1: 评估单个模型

```bash
# 激活 UATR 环境
conda activate UATR

# 使用 Python 脚本
python evaluate_student_deepship.py \
    --checkpoint "/media/hdd1/fubohan/Code/UATR/checkpoints/ablation_deepship/num_epochs200_weight_decay8e-05_batch_size16_lr0.0004_distill_typeMTSKD_Temp_freezeTrue_USE_DYNAMIC_DISTILL_WEIGHTFalse_p_encoder0.1_p_classifier0.35/best_student.pth" \
    --config "/media/hdd1/fubohan/Code/UATR/configs/train_distillation_deepship.yaml" \
    --batch_size 32 \
    --gpu 0 \
    --dataset test

# 或使用 shell 脚本
./evaluate_single.sh \
    "/media/hdd1/fubohan/Code/UATR/checkpoints/ablation_deepship/num_epochs200_weight_decay8e-05_batch_size16_lr0.0004_distill_typeMTSKD_Temp_freezeTrue_USE_DYNAMIC_DISTILL_WEIGHTFalse_p_encoder0.1_p_classifier0.35/best_student.pth" \
    0 test
```

### 方式 2: 批量评估所有模型

```bash
# 激活 UATR 环境
conda activate UATR

# 使用 Python 脚本
python batch_evaluate_ablation_deepship.py

# 或使用 shell 脚本
./evaluate_all_ablation.sh
```

## 已测试

✅ 已成功测试单个模型评估,结果显示:
- **Accuracy**: 86.9827%
- **F1-Macro**: 86.9924%
- **AUC-OVO**: 97.7024%
- 所有指标正常计算

✅ 已确认 `checkpoints/ablation_deepship/` 目录下有 4 个模型可供评估:
1. MemKD
2. Tser
3. baseline (无蒸馏)
4. MTSKD_Temp

## 环境配置

所有依赖已安装在 `UATR` conda 环境中:
- PyTorch 2.6.0
- torchaudio 2.6.0
- torchvision 0.21.0
- transformers 4.57.3
- scikit-learn 1.7.2
- pandas 2.3.3

## 评估指标说明

### Accuracy (准确率)
整体分类正确的样本比例

### F1-Score
- **Macro**: 各类别 F1 的算术平均(不考虑类别不平衡)
- **Micro**: 基于所有样本的总体指标计算 F1
- **Weighted**: 按类别样本数加权计算 F1

### AUC (Area Under Curve)
- **OVO (One-vs-One)**: 一对一策略的 AUC
- **OVR (One-vs-Rest)**: 一对多策略的 AUC

### Confusion Matrix (混淆矩阵)
显示各类别的预测混淆情况

### Per-class Metrics
每个类别的 Precision (精确率)、Recall (召回率)、F1-Score

## 输出文件位置

### 单模型评估
- 结果文件: `{checkpoint目录}/evaluation_results_{dataset}.txt`
- 包含: 所有指标、混淆矩阵、分类报告

### 批量评估
- 汇总文件: `{base_dir}/evaluation_summary_{dataset}.txt`
- 各模型结果: `{每个模型目录}/evaluation_results_{dataset}.txt`

## 下一步建议

1. **批量评估**: 运行 `./evaluate_all_ablation.sh` 评估所有 4 个模型
2. **比较结果**: 查看汇总文件,比较不同蒸馏方法的性能
3. **分析指标**: 重点关注 Accuracy、F1-Score 和 AUC 的综合表现
4. **保存报告**: 将评估结果整理成表格或图表,用于论文/报告

## 注意事项

1. 确保在 `UATR` 环境中运行
2. 确保数据集路径在配置文件中正确设置
3. 如遇 GPU 显存不足,可减小 `--batch_size`
4. 批量评估可能需要较长时间(约 20-40 分钟,取决于模型数量和数据集大小)

## 问题反馈

如有任何问题或需要修改,请告知!
