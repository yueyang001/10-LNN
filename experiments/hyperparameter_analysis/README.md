# MTSKD_Temp 超参数分析

本目录用于组织最终 `MTSKD_Temp` 算法的超参数实验。四组实验分别验证：

- `01_beta_analysis`：固定 `beta` 与可学习 `beta`
- `02_mtskd_weight_analysis`：固定融合比例与可学习融合权重
- `03_memkd_weight_analysis`：MemKD 短期/长期损失权重
- `04_z_random_range_analysis`：长期记忆偏移量 `z_random` 的采样范围
- `run_plan.py`：通用实验启动脚本

旧的消融实验目录仍保留在仓库中，作为历史实验计划参考；新的论文超参数分析建议使用以上四个目录。

## 数据集选择

每个 `plan.yaml` 默认仍是 DeepShip：

```yaml
dataset: DeepShip
base_config: configs/train_distillation_deepship.yaml
train_script: train_distillation_deepship.py
```

现在可以在运行时通过 `--dataset` 覆盖数据集，不需要复制四套 `plan.yaml`。

DeepShip 默认运行：

```bash
python experiments/hyperparameter_analysis/run_plan.py \
  --plan experiments/hyperparameter_analysis/01_beta_analysis/plan.yaml \
  --gpus 4,5,6,7
```

ShipSEAR 运行：

```bash
python experiments/hyperparameter_analysis/run_plan.py \
  --plan experiments/hyperparameter_analysis/01_beta_analysis/plan.yaml \
  --dataset shipsear \
  --gpus 4,5,6,7
```

`--dataset shipsear` 会自动切换为：

- 基础配置：`configs/train_distillation_shipsear.yaml`
- 训练脚本：`train_distillation_shipsear.py`
- 固定超参：`batch_size=20`、`lr=0.0005`、`weight_decay=0.0001`、`p_encoder=0.355`、`p_classifier=0.255`

DeepShip 的固定超参仍为：`batch_size=16`、`lr=0.0004`、`weight_decay=0.00008`、`p_encoder=0.1`、`p_classifier=0.35`。

## 运行方式

先检查命令和生成配置：

```bash
python experiments/hyperparameter_analysis/run_plan.py \
  --plan experiments/hyperparameter_analysis/01_beta_analysis/plan.yaml \
  --dry-run
```

正式运行：

```bash
python experiments/hyperparameter_analysis/run_plan.py \
  --plan experiments/hyperparameter_analysis/01_beta_analysis/plan.yaml \
  --gpus 4,5,6,7
```

其他实验只需要替换 `--plan` 路径。

## 并行端口

训练脚本会从生成后的 `config.yaml` 读取 `distributed.master_port`。如果同时启动多个超参数实验，原来多个进程会使用同一个端口，容易出现分布式初始化冲突。

`run_plan.py` 现在默认使用 `--master-port auto`，会根据数据集、实验层级和实验编号给每个实验写入不同的端口，并跳过当前已占用端口。因此四个实验目录可以同时启动：

```bash
python experiments/hyperparameter_analysis/run_plan.py --plan experiments/hyperparameter_analysis/01_beta_analysis/plan.yaml --gpus 4,5,6,7
python experiments/hyperparameter_analysis/run_plan.py --plan experiments/hyperparameter_analysis/02_mtskd_weight_analysis/plan.yaml --gpus 0,1,2,3
```

如果确实需要指定固定端口，可以显式传入：

```bash
python experiments/hyperparameter_analysis/run_plan.py \
  --plan experiments/hyperparameter_analysis/01_beta_analysis/plan.yaml \
  --master-port 12356 \
  --gpus 4,5,6,7
```

并行运行时建议保留默认 `auto`。

## 输出约定

DeepShip 默认输出路径保持不变：

- 临时训练配置：`checkpoints/hyperparameter_analysis/<level>/<experiment_name>/config.yaml`
- 训练结果：`checkpoints/hyperparameter_analysis/<level>/<experiment_name>/`
- 汇总 CSV：`results/hyperparameter_analysis/<level>/results.csv`
- 完成标记：每个实验成功后写入 `finished.flag`

使用 `--dataset shipsear` 时，为避免和 DeepShip 结果混在一起，会输出到数据集子目录：

- 临时训练配置：`checkpoints/hyperparameter_analysis/shipsear/<level>/<experiment_name>/config.yaml`
- 训练结果：`checkpoints/hyperparameter_analysis/shipsear/<level>/<experiment_name>/`
- 汇总 CSV：`results/hyperparameter_analysis/shipsear/<level>/results.csv`

## 论文说明建议

`beta` 和 `mtskd_weight` 推荐同时报告固定权重与可学习权重。即使可学习权重不是所有单项指标的绝对最高，也可以强调其在无需人工搜索的情况下取得稳定且具有竞争力的性能。

`memkd_short_weight / memkd_long_weight` 和 `z_random_range` 则作为结构性超参数，用于证明默认设计在短期状态变化与长期记忆建模之间取得了更好的平衡。
