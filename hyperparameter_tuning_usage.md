# 超参数分析脚本使用指南

## ✅ 已完成的功能

### 1. 日志输出（与 train_distillation_shipsear.py 一致）
- ✅ **训练过程中**：每 20 个 batch 输出一次日志
  ```
  Epoch [1] Batch [0/100] Loss: 0.1234, Hard: 0.0234, Soft: 0.1000 Acc: 85.50% LR: 0.001000 Alpha: 0.5000 Beta: 0.5000 MemKD: 0.5000
  ```

- ✅ **每个 epoch 结束后**：输出训练和验证指标
  ```
  Epoch [1/100] Train Loss: 0.2345, Hard: 0.0345, Soft: 0.2000 Train Acc: 88.23%, Val Loss: 0.2100, Val Acc: 89.50%, Best Val Acc: 89.50%, LR: 0.001000
  ```

### 2. 支持的日志特性
- ✅ 动态蒸馏权重通知（`set_epoch`）
- ✅ Loss 异常值检测和警告
- ✅ 训练过程中的实时损失和准确率显示
- ✅ Alpha、Beta、MemKD 权重实时显示
- ✅ 学习率实时显示

## 📊 日志文件结构

运行后会在保存目录生成：
```
checkpoints/ShipsEar/hyperparameter_tuning/hyperparam_search_YYYYMMDD_HHMMSS/
├── hyperparam_tuning_YYYYMMDD_HHMMSS.log  # 完整日志文件
├── step_a_lambda_m_scan.csv               # Step A 结果
├── step_b_beta_scan.csv                   # Step B 结果
└── hyperparameter_tuning_summary.csv      # 汇总结果
```

## 🚀 使用方式

### 方式 1：使用 YAML 配置（默认 100 epoch）
```bash
python hyperparameter_tuning.py
```

### 方式 2：自定义训练轮数
```bash
python hyperparameter_tuning.py --num_epochs 20
```

### 方式 3：完全自定义
```bash
python hyperparameter_tuning.py \
  --config configs/hyperparameter_tuning.yaml \
  --gpus "4,5,6,7" \
  --lambda_m_values "0.0,0.25,0.5,0.75,1.0" \
  --beta_values "0.0,0.25,0.5,0.75,1.0" \
  --fixed_beta 0.5 \
  --num_epochs 50
```

## 📝 日志输出示例

### 训练开始
```
============================================================
超参数分析开始
============================================================
配置: configs/hyperparameter_tuning.yaml
GPUs: [4, 5, 6, 7]
World size: 4
Step A - 扫描 lambda_m: 0.0,0.25,0.5,0.75,1.0
Step B - 扫描 beta: 0.0,0.25,0.5,0.75,1.0
每轮训练 epochs: 100
固定 beta (Step A): 0.5
============================================================
```

### Step A: 扫描 lambda_m
```
############################################################
Step A: 扫描 lambda_m (固定 beta=0.5)
############################################################

============================================================
开始训练: lambda_m=0.0, beta=0.5
训练轮数: 100
============================================================
Epoch [1] Batch [0/100] Loss: 2.3456, Hard: 1.2345, Soft: 1.1111 Acc: 45.20% LR: 0.001000 Alpha: 0.5000 Beta: 0.5000 MemKD: 0.0000
Epoch [1] Batch [20/100] Loss: 1.8765, Hard: 0.9876, Soft: 0.8889 Acc: 52.30% LR: 0.001000 Alpha: 0.5000 Beta: 0.5000 MemKD: 0.0000
...
Epoch [1/100] Train Loss: 1.5432, Hard: 0.7654, Soft: 0.7778 Train Acc: 55.45%, Val Loss: 1.4321, Val Acc: 58.23%, Best Val Acc: 58.23%, LR: 0.001000
Epoch [2/100] Train Loss: 1.2345, Hard: 0.5432, Soft: 0.6913 Train Acc: 62.34%, Val Loss: 1.1234, Val Acc: 65.12%, Best Val Acc: 65.12%, LR: 0.001000
...
✅ 训练完成! lambda_m=0.0, beta=0.5, 最佳验证准确率: 88.50%
============================================================

lambda_m=0.00, beta=0.50 -> Val Acc: 88.50%
```

### Step B: 扫描 beta
```
############################################################
Step B: 扫描 beta (固定 lambda_m=0.5)
############################################################

============================================================
开始训练: lambda_m=0.5, beta=0.0
训练轮数: 100
============================================================
Epoch [1/100] Train Loss: 1.4567, Hard: 0.6789, Soft: 0.7778 Train Acc: 58.90%, Val Loss: 1.3456, Val Acc: 62.45%, Best Val Acc: 62.45%, LR: 0.001000
...
✅ 训练完成! lambda_m=0.5, beta=0.0, 最佳验证准确率: 90.23%
============================================================

lambda_m=0.50, beta=0.00 -> Val Acc: 90.23%
```

### 最终结果
```
============================================================
超参数分析完成!
============================================================
最佳配置: lambda_m=0.50, beta=0.00
最高准确率: 90.23%
============================================================
```

## 🔍 日志关键字段说明

### 训练过程中（每个 batch）
- `Loss`: 总损失（Hard Loss + Soft Loss）
- `Hard`: 硬标签损失（CrossEntropy）
- `Soft`: 软标签损失（蒸馏损失）
- `Acc`: 当前批次的训练准确率
- `LR`: 学习率
- `Alpha`: 蒸馏权重（动态调整）
- `Beta`: 时序损失和最终损失的权重
- `MemKD`: MemKD 权重

### 每个 epoch 结束
- `Train Loss`: 训练集平均损失
- `Train Acc`: 训练集准确率
- `Val Loss`: 验证集损失
- `Val Acc`: 验证集准确率
- `Best Val Acc`: 历史最佳验证准确率

## 💡 使用建议

### 快速测试（20 epoch）
```bash
python hyperparameter_tuning.py --num_epochs 20
```
- 适合快速验证代码和配置
- 耗时较短

### 正式实验（100 epoch）
```bash
python hyperparameter_tuning.py
```
- 使用 YAML 配置中的 epoch 数量
- 获得更准确的结果

### 监控训练
实时查看日志文件：
```bash
tail -f checkpoints/ShipsEar/hyperparameter_tuning/hyperparam_search_*/hyperparam_tuning_*.log
```

## ⚙️ 配置文件说明

`configs/hyperparameter_tuning.yaml` 中的关键配置：

```yaml
training:
  log_interval: 20  # 训练过程中的日志输出间隔（batch数）
  num_epochs: 100   # 训练轮数

distillation:
  USE_DYNAMIC_DISTILL_WEIGHT: True  # 启用动态蒸馏权重
```

## ✨ 与 train_distillation_shipsear.py 的对比

| 特性 | train_distillation_shipsear.py | hyperparameter_tuning.py |
|------|------------------------------|-------------------------|
| 训练过程日志 | ✅ 每 20 batch | ✅ 每 20 batch |
| Epoch 日志 | ✅ 每个 epoch | ✅ 每个 epoch |
| Alpha/Beta/MemKD 显示 | ✅ | ✅ |
| 动态蒸馏权重 | ✅ | ✅ |
| 异常值检测 | ✅ | ✅ |
| 梯度裁剪 | ✅ | ✅ |
| 学习率调度 | ✅ | ✅ |
| 保存模型 | ✅ | ⚠️ 可选 |
| 超参数扫描 | ❌ | ✅ |

## 🎯 总结

现在的 `hyperparameter_tuning.py` 已经完全实现了与 `train_distillation_shipsear.py` 一致的日志输出：
- ✅ 每个 batch 的详细信息
- ✅ 每个 epoch 的训练和验证指标
- ✅ 实时显示所有关键超参数
- ✅ 支持动态蒸馏权重
- ✅ 异常值检测和警告

你现在可以直接运行 `python hyperparameter_tuning.py`，享受详细的日志输出！🎉
