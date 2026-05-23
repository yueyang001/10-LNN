# MTSKD_Temp 超参数分析

本目录现在围绕最终采用的 `MTSKD_Temp` 算法组织超参数实验。四组实验分别验证：`beta`、MemKD/KL_Temp 融合权重、MemKD 长短期权重、`z_random` 采样范围。

## 目录结构

- `01_beta_analysis`: 分析固定 `beta` 与可学习 `beta`
- `02_mtskd_weight_analysis`: 分析固定融合比例与可学习融合权重
- `03_memkd_weight_analysis`: 分析 MemKD 短期/长期损失权重
- `04_z_random_range_analysis`: 分析长期记忆偏移量采样范围
- `run_plan.py`: 通用实验启动脚本

旧的三类消融目录仍保留在仓库中，作为历史实验计划参考；新的论文超参数分析建议使用以上四个目录。

## 运行方式

先检查命令：

```bash
python experiments/hyperparameter_analysis/run_plan.py --plan experiments/hyperparameter_analysis/01_beta_analysis/plan.yaml --dry-run
```

正式运行：

```bash
python experiments/hyperparameter_analysis/run_plan.py --plan experiments/hyperparameter_analysis/01_beta_analysis/plan.yaml --gpus 4,5,6,7
```

其他实验只需要替换 `--plan` 路径。

## 输出约定

- 临时训练配置：`checkpoints/hyperparameter_analysis/<level>/<experiment_name>/config.yaml`
- 训练结果：`checkpoints/hyperparameter_analysis/<level>/<experiment_name>/`
- 汇总 CSV：`results/hyperparameter_analysis/<level>/results.csv`
- 完成标记：每个实验成功后写入 `finished.flag`

## 论文说明建议

`beta` 和 `mtskd_weight` 推荐同时报告固定权重与可学习权重。即使可学习权重不是所有单项指标的绝对最高，也可以强调其在无需人工搜索的情况下取得稳定且具有竞争力的性能。

`memkd_short_weight / memkd_long_weight` 和 `z_random_range` 则作为结构性超参数，用于证明默认设计在短期状态变化与长期记忆建模之间取得更好的平衡。
