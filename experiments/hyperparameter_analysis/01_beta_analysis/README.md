# beta 超参数分析

本实验用于分析 `MTSKD_Temp` 中时序 KL 损失与最终输出 KL 损失的融合系数：

```python
kl_loss = beta * seq_loss + (1 - beta) * final_loss
```

实验包含两类设置：

- 固定 `beta`: `0.1 / 0.3 / 0.5 / 0.7 / 0.9`
- 可学习 `beta`: 保持当前算法的自适应权重设计

论文表述建议：固定 `beta` 用于展示不同人工权重的敏感性；可学习 `beta` 用于证明方法可以减少人工搜索，并在合理范围内保持稳定性能。

运行：

```bash
python experiments/hyperparameter_analysis/run_plan.py --plan experiments/hyperparameter_analysis/01_beta_analysis/plan.yaml --gpus 4,5,6,7
```
