# MTSKD 融合权重分析

本实验用于分析 `MTSKD_Temp` 中 MemKD 与 KL_Temp 的融合比例：

```python
soft_loss = memkd_weight * memkd_loss + (1 - memkd_weight) * kl_loss
```

实验包含：

- `0.2 MemKD + 0.8 KL_Temp`
- `0.5 MemKD + 0.5 KL_Temp`
- `0.8 MemKD + 0.2 KL_Temp`
- 可学习融合权重

论文表述建议：固定比例用于展示融合权重敏感性；可学习融合权重用于证明算法不依赖人工指定单一比例，能够自适应平衡记忆蒸馏与标准化 KL 蒸馏。

运行：

```bash
python experiments/hyperparameter_analysis/run_plan.py --plan experiments/hyperparameter_analysis/02_mtskd_weight_analysis/plan.yaml --gpus 4,5,6,7
```
