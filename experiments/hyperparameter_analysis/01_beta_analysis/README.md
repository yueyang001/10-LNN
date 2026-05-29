# beta 超参数分析

本实验用于分析 `MTSKD_Temp` 中时序 KL 损失与最终输出 KL 损失的融合系数：

```python
kl_loss = beta * seq_loss + (1 - beta) * final_loss
```

当前实验设置按照会议意见调整为：`beta` 始终作为可学习参数，分别测试不同初始值对整体网络训练的影响。

- 可学习 `beta` 初始值：`0.1 / 0.3 / 0.5 / 0.7 / 0.9`
- 每组实验均设置 `distillation.learnable_beta: true`
- 旧版“固定 beta 扫描 + 单独可学习 beta”实验已在 `plan.yaml` 底部注释备份，当前不启用

论文表述建议：该实验用于说明可学习融合系数对不同初始化的敏感性，以及模型是否能够在不同初始权重下稳定收敛到具有竞争力的性能。

运行：

```bash
python experiments/hyperparameter_analysis/run_plan.py --plan experiments/hyperparameter_analysis/01_beta_analysis/plan.yaml --gpus 4,5,6,7
```
