# 🎯 多组K-Fold划分 - 完整使用指南

## 📋 概述

为了选择**最优的K-Fold划分**进行最终训练，我创建了一套完整的多划分对比工具：

1. **生成多组划分** - 使用不同随机种子生成多组K-Fold
2. **分别训练各组** - 逐组运行完整的十折叠训练
3. **对比分析结果** - 自动对比各组的性能
4. **选择最优划分** - 推荐性能最好的划分

---

## 🚀 快速开始

### 方式1️⃣：完整流程（一键执行，推荐）

```bash
python kfold_multi_train.py --all \
    --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622 \
    --num-splits 3 \
    --gpus 4,5,6,7
```

**这会自动执行：**
1. ✓ 生成3组K-Fold划分（种子: 42, 123, 456）
2. ✓ 运行每组的10折叠训练
3. ✓ 对比分析所有结果
4. ✓ 推荐最优划分

---

## 📂 分步使用方法

### 第1步：生成多组划分

生成3组不同的K-Fold划分：

```bash
python kfold_multi_split_generator.py \
    --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622 \
    --output-dir results/kfold_splits_multi \
    --num-splits 3
```

**或使用--all中的--generate选项：**

```bash
python kfold_multi_train.py --generate \
    --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622 \
    --num-splits 3
```

**输出：**
```
results/kfold_splits_multi/
├── split_00_seed42/           # 第1组（种子42）
│   ├── kfold_summary.txt
│   ├── kfold_fold00.txt
│   ├── kfold_fold01.txt
│   └── ...
│
├── split_01_seed123/          # 第2组（种子123）
│   ├── kfold_summary.txt
│   └── ...
│
├── split_02_seed456/          # 第3组（种子456）
│   ├── kfold_summary.txt
│   └── ...
│
└── splits_comparison.txt       # 划分对比报告
```

### 第2步：分别训练各组划分

```bash
# 方式A：依次训练所有split
python kfold_multi_train.py --train \
    --output-dir results \
    --gpus 4,5,6,7

# 方式B：或使用--all中的--train选项
python kfold_multi_train.py --all --train
```

**或逐个手动训练：**

```bash
# 训练 split_00 (seed=42)
python kfold_shipsear_integration.py --train-all \
    --gpus 4,5,6,7 \
    --splits-dir results/kfold_splits_multi/split_00_seed42

# 训练 split_01 (seed=123)
python kfold_shipsear_integration.py --train-all \
    --gpus 4,5,6,7 \
    --splits-dir results/kfold_splits_multi/split_01_seed123

# 训练 split_02 (seed=456)
python kfold_shipsear_integration.py --train-all \
    --gpus 4,5,6,7 \
    --splits-dir results/kfold_splits_multi/split_02_seed456
```

**或并行训练（更快）：**

```bash
# 同时训练所有split（假设GPU充足）
for split in results/kfold_splits_multi/split_*; do
    python kfold_shipsear_integration.py --train-all \
        --gpus 4,5,6,7 \
        --splits-dir "$split" &
done
wait
```

**输出：**
```
results/kfold_cv_shipsear_multi/
├── split_00_seed42/
│   ├── kfold_shipsear_results.csv    # split_00的结果
│   ├── training_summary.txt
│   └── fold_00/ ... fold_09/         # 各Fold的检查点
│
├── split_01_seed123/
│   ├── kfold_shipsear_results.csv    # split_01的结果
│   └── ...
│
├── split_02_seed456/
│   ├── kfold_shipsear_results.csv    # split_02的结果
│   └── ...
│
└── comparison_report.txt             # 最终对比报告 ⭐
```

### 第3步：对比分析结果

```bash
python kfold_comparison_analyzer.py \
    --results-dir results/kfold_cv_shipsear_multi
```

**或查看生成的报告文件：**

```bash
cat results/kfold_cv_shipsear_multi/comparison_report.txt
```

---

## 📊 对比分析示例

当对比分析完成时，你会看到类似这样的输出：

```
================================================================================
K-Fold划分结果对比分析
================================================================================

【各划分性能统计】

排名  划分名称                 平均精度    中位数      标准差      范围
------                                                               
1     split_00_seed42          0.9167      0.9200      0.0089      [0.8900, 0.9400] ⭐
2     split_01_seed123         0.9112      0.9150      0.0145      [0.8800, 0.9350]
3     split_02_seed456         0.9045      0.9050      0.0203      [0.8700, 0.9250]

【详细精度对比】

1. split_00_seed42
   平均精度: 0.9167
   中位数:   0.9200
   标准差:   0.0089
   最高精度: 0.9400
   最低精度: 0.8900
   Fold精度列表: 0.8900, 0.9100, 0.9150, 0.9200, 0.9200, 0.9250, 0.9300, 0.9350, 0.9400, 0.9200

【推荐】

✓ 推荐使用: split_00_seed42
  理由:
  - 平均精度最高: 0.9167
  - 标准差: 0.0089
  - 性能稳定，Fold间差异小
```

---

## 🎯 自定义随机种子

如果想指定特定的随机种子：

```bash
python kfold_multi_split_generator.py \
    --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622 \
    --output-dir results/kfold_splits_multi \
    --num-splits 5 \
    --seeds 42 100 200 300 400
```

---

## 📈 完整工作流程（示例）

```bash
# 【步骤1】生成3组划分（15-30分钟）
python kfold_multi_split_generator.py \
    --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622 \
    --output-dir results/kfold_splits_multi \
    --num-splits 3

# 【步骤2】训练所有split（10-20小时，取决于GPU）
# 方式A：顺序训练（推荐，占用1组GPU）
for split in results/kfold_splits_multi/split_*/; do
    echo "Training $split..."
    python kfold_shipsear_integration.py --train-all \
        --gpus 4,5,6,7 \
        --splits-dir "$split"
done

# 方式B：并行训练（如果有多组GPU）
# for split in results/kfold_splits_multi/split_*/; do
#     python kfold_shipsear_integration.py --train-all \
#         --gpus 4,5,6,7 \
#         --splits-dir "$split" &
# done
# wait

# 【步骤3】对比分析结果（5分钟）
python kfold_comparison_analyzer.py \
    --results-dir results/kfold_cv_shipsear_multi

# 【步骤4】查看详细报告
cat results/kfold_cv_shipsear_multi/comparison_report.txt

# 【步骤5】使用最优划分进行最终训练（可选）
# 假设 split_00_seed42 是最优的
python kfold_shipsear_integration.py --train-all \
    --gpus 4,5,6,7 \
    --splits-dir results/kfold_splits_multi/split_00_seed42 \
    --results-dir results/kfold_cv_shipsear_final
```

---

## 🔍 查看对比结果

### 查看划分对比
```bash
cat results/kfold_splits_multi/splits_comparison.txt
```

### 查看性能对比
```bash
cat results/kfold_cv_shipsear_multi/comparison_report.txt
```

### 查看特定split的详细结果
```bash
cat results/kfold_cv_shipsear_multi/split_00_seed42/kfold_shipsear_results.csv
```

### 用Python分析
```python
import pandas as pd

# 对比所有split的平均精度
splits = ['split_00_seed42', 'split_01_seed123', 'split_02_seed456']

for split in splits:
    df = pd.read_csv(f'results/kfold_cv_shipsear_multi/{split}/kfold_shipsear_results.csv')
    mean_acc = df['best_acc'].mean()
    std_acc = df['best_acc'].std()
    print(f"{split}: {mean_acc:.4f} ± {std_acc:.4f}")
```

---

## 📝 生成的文件说明

```
results/
├── kfold_splits_multi/
│   ├── split_00_seed42/
│   │   ├── kfold_summary.txt           # split_00的划分汇总
│   │   ├── kfold_fold00.txt ~ fold09.txt
│   │   └── kfold_indices.txt
│   │
│   ├── split_01_seed123/
│   ├── split_02_seed456/
│   │
│   ├── splits_summary.json             # 所有split的元数据
│   └── splits_comparison.txt           # 划分对比报告
│
└── kfold_cv_shipsear_multi/
    ├── split_00_seed42/
    │   ├── kfold_shipsear_results.csv   # 结果CSV
    │   ├── training_summary.txt         # 训练报告
    │   └── fold_00/ ~ fold_09/         # 各Fold的模型检查点
    │
    ├── split_01_seed123/
    ├── split_02_seed456/
    │
    └── comparison_report.txt           # ⭐ 最终对比报告
```

---

## 💡 使用场景

### 场景1：初始研究探索
```bash
# 生成3组划分，快速对比
python kfold_multi_train.py --all \
    --num-splits 3 \
    --gpus 4,5,6,7
```

### 场景2：论文实验
```bash
# 生成5组划分，获得更鲁棒的结果
python kfold_multi_train.py --all \
    --num-splits 5 \
    --seeds 42 123 456 789 999 \
    --gpus 4,5,6,7

# 取性能最好和最稳定的划分作为论文的主要结果
# 其他结果作为补充实验
```

### 场景3：调试和优化
```bash
# 生成2组快速测试
python kfold_multi_train.py --all \
    --num-splits 2 \
    --gpus 4,5,6,7

# 基于结果调整模型配置
# 然后用最优划分重新训练
```

---

## 📊 推荐配置

### 小规模实验（快速）
```bash
--num-splits 2          # 2组划分
--seeds 42 123          # 两个不同的种子
# 总运行时间: ~2-3小时 (3组GPU) 或 ~6-8小时 (1组GPU)
```

### 中等规模实验
```bash
--num-splits 3          # 3组划分（默认）
--seeds 42 123 456      # 三个不同的种子
# 总运行时间: ~3-5小时 (3组GPU) 或 ~10-15小时 (1组GPU)
```

### 大规模实验（详细对比）
```bash
--num-splits 5          # 5组划分
--seeds 42 123 456 789 999
# 总运行时间: ~5-8小时 (3组GPU) 或 ~20-30小时 (1组GPU)
```

---

## ✅ 工作流检查清单

- [ ] 已确认数据目录正确
- [ ] 已安装所有依赖
- [ ] 已决定要生成多少组划分（推荐3-5组）
- [ ] 已生成多组K-Fold划分
- [ ] 已完成所有split的训练
- [ ] 已查看对比分析结果
- [ ] 已识别出最优划分
- [ ] 已用最优划分进行最终训练（可选）
- [ ] 已保存结果到git
- [ ] 已在论文中记录使用的划分配置

---

## 🎓 最佳实践

### ✅ 推荐做法
1. **生成多组划分** - 使用至少3个不同的种子
2. **对比分析结果** - 选择平均精度高且标准差小的划分
3. **记录配置** - 在论文中说明使用的随机种子
4. **版本控制** - 保存所有划分和结果

### ❌ 避免做法
1. 只用单一种子进行划分（容易偏差）
2. 不对比不同划分的结果（可能遗漏更优方案）
3. 忘记记录使用的种子和配置（不可复现）
4. 不保存到版本控制（丢失历史记录）

---

## 🔗 相关命令速查

```bash
# 【生成多组划分】
python kfold_multi_split_generator.py --data-dir <path> --num-splits 3

# 【训练所有split】
python kfold_multi_train.py --train --gpus 4,5,6,7

# 【对比分析】
python kfold_comparison_analyzer.py --results-dir results/kfold_cv_shipsear_multi

# 【查看报告】
cat results/kfold_cv_shipsear_multi/comparison_report.txt

# 【使用最优划分训练】
python kfold_shipsear_integration.py --train-all \
    --gpus 4,5,6,7 \
    --splits-dir results/kfold_splits_multi/split_00_seed42

# 【完整流程一键执行】
python kfold_multi_train.py --all --data-dir <path> --num-splits 3 --gpus 4,5,6,7
```

---

## 📞 获取帮助

查看脚本的帮助信息：

```bash
python kfold_multi_split_generator.py --help
python kfold_comparison_analyzer.py --help
python kfold_multi_train.py --help
```

---

## 🎉 总结

现在你拥有了完整的多划分对比系统，可以：

✅ 生成多组不同的K-Fold划分  
✅ 对每组划分运行完整的十折叠训练  
✅ 自动对比分析所有结果  
✅ 推荐性能最优的划分  
✅ 用最优划分进行最终训练  

**立即开始：**

```bash
python kfold_multi_train.py --all \
    --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622 \
    --num-splits 3 \
    --gpus 4,5,6,7
```

祝你找到最优的划分！🚀
