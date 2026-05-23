# z_random 采样范围分析

本实验用于分析 MemKD 长期记忆偏移量 `z_random` 的采样范围。

候选范围：

- `quarter`: `randint(2, T_max // 4)`
- `half`: `randint(2, T_max // 2)`
- `upper_half`: `randint(T_max // 4, T_max // 2)`

默认推荐配置是 `half`，因为它同时覆盖短期与中长期状态差异，通常比只关注局部或只关注远期更均衡。

运行：

```bash
python experiments/hyperparameter_analysis/run_plan.py --plan experiments/hyperparameter_analysis/04_z_random_range_analysis/plan.yaml --gpus 4,5,6,7
```
