# 学生网络消融实验说明

本组实验针对 `models/LNN.py` 中的学生网络 `AudioCfC`，重点分析 `BiParallelCrossSliceCfC` 和 `DRASP`。

## 实验目录

| 目录 | 实验内容 |
|---|---|
| `05_student_cfc_branch_ablation` | 四个 CfC 分支消融：完整四分支、Only FL、FL+FG、FL+BL |
| `06_student_temporal_granularity` | 时间切片粒度分析：`window_size` 与 `drasp_segment_len` 同步取 `1/2/4/8/16` |
| `07_student_cfc_capacity` | CfC 容量分析：`wiring_units` 取 `80/96/128/160/192` |
| `08_student_drasp_ablation` | DRASP 池化消融：只用 global ASP、只用 local ASP、完整 DRASP |

## 单个实验运行

先 dry-run 检查命令和生成配置：

```bash
python experiments/hyperparameter_analysis/run_plan.py \
  --plan experiments/hyperparameter_analysis/05_student_cfc_branch_ablation/plan.yaml \
  --dataset deepship \
  --gpus 4,5,6,7 \
  --dry-run
```

正式运行 DeepShip：

```bash
python experiments/hyperparameter_analysis/run_plan.py \
  --plan experiments/hyperparameter_analysis/05_student_cfc_branch_ablation/plan.yaml \
  --dataset deepship \
  --gpus 4,5,6,7
```

正式运行 ShipSEAR：

```bash
python experiments/hyperparameter_analysis/run_plan.py \
  --plan experiments/hyperparameter_analysis/05_student_cfc_branch_ablation/plan.yaml \
  --dataset shipsear \
  --gpus 4,5,6,7
```

## 按数据集运行四组实验

使用仓库根目录下的脚本，每次启动一个数据集的四组实验：

```bash
DATASET=deepship bash run_student_ablation_exp.sh
DATASET=shipsear bash run_student_ablation_exp.sh
```

常用环境变量：

```bash
DATASET=shipsear DRY_RUN=1 bash run_student_ablation_exp.sh
DATASET=deepship CONDA_ENV=UATR bash run_student_ablation_exp.sh
DATASET=shipsear GPU_BRANCH=4,5,6,7 GPU_TEMPORAL=4,5,6,7 bash run_student_ablation_exp.sh
```

脚本默认创建 `student_network_ablation_deepship` 或
`student_network_ablation_shipsear` 会话，因此两个数据集可以分时启动，
也可以在 GPU 资源足够时分别挂在不同会话中运行。

## 输出位置

DeepShip 默认输出：

```text
checkpoints/hyperparameter_analysis/<实验目录>/<实验名>/
results/hyperparameter_analysis/<实验目录>/results.csv
```

ShipSEAR 默认输出：

```text
checkpoints/hyperparameter_analysis/shipsear/<实验目录>/<实验名>/
results/hyperparameter_analysis/shipsear/<实验目录>/results.csv
```

`results.csv` 会记录 `best_acc`、`best_auc`、`best_f1` 和对应 checkpoint 目录。
