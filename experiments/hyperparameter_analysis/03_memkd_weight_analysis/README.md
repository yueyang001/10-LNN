# MemKD 长短期权重分析

本实验用于分析 MemKD 内部短期记忆与长期记忆的损失权重：

```python
memkd_loss = short_weight * memkd_loss_short + long_weight * memkd_loss_long
```

候选范围：

- `short_weight = 0.25 / 0.5 / 1.0`
- `long_weight = 0.5 / 1.0 / 2.0`

默认推荐配置是 `short_weight=0.5, long_weight=1.0`，用于平衡局部状态变化与较长时间跨度的记忆差异。

运行：

```bash
python experiments/hyperparameter_analysis/run_plan.py --plan experiments/hyperparameter_analysis/03_memkd_weight_analysis/plan.yaml --gpus 4,5,6,7
```
