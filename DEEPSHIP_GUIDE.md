# 🎯 DeepShip K-Fold十折叠交叉验证 - 完整使用指南

## 📋 新增的DeepShip专用脚本

为了与ShipEar保持一致，我为**DeepShip数据集**创建了对应的脚本：

### 核心脚本（2个）

1. **`kfold_deepship_integration.py`** (400行)
   - DeepShip集成脚本（对应ShipEar的`kfold_shipsear_integration.py`）
   - 支持单个Fold或批量训练
   - 自动收集结果统计

2. **`kfold_multi_train_deepship.py`** (400行)
   - DeepShip多组K-Fold训练脚本（对应ShipEar的`kfold_multi_train.py`）
   - 完整流程：生成 → 训练 → 对比
   - 自动化程度高

---

## 🚀 快速开始

### 方法1️⃣：完整流程（推荐一键执行）

```bash
python kfold_multi_train_deepship.py --all \
    --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/DeepShip_622 \
    --num-splits 3 \
    --gpus 4,5,6,7
```

**自动执行：**
1. ✓ 生成3组K-Fold划分（种子: 42, 123, 456）
2. ✓ 对每组运行完整的10折叠训练
3. ✓ 对比分析所有结果
4. ✓ 推荐最优划分

### 方法2️⃣：分步执行

```bash
# 第1步：生成划分
python kfold_multi_train_deepship.py --generate \
    --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/DeepShip_622 \
    --num-splits 3

# 第2步：训练所有split
python kfold_multi_train_deepship.py --train \
    --gpus 4,5,6,7

# 第3步：对比分析
python kfold_multi_train_deepship.py --compare \
    --results-dir results/kfold_cv_deepship_multi
```

### 方法3️⃣：训练单个Fold（测试）

```bash
python kfold_deepship_integration.py --fold 0 \
    --gpus 4,5,6,7 \
    --splits-dir results/kfold_splits
```

### 方法4️⃣：训练所有Fold（已有划分时）

```bash
python kfold_deepship_integration.py --all \
    --gpus 4,5,6,7 \
    --splits-dir results/kfold_splits
```

---

## 📂 目录结构对比

### ShipEar（已有）
```
results/
├── kfold_splits/              # K-Fold划分
├── kfold_cv_shipsear/         # 训练结果
├── kfold_splits_multi/        # 多组划分
└── kfold_cv_shipsear_multi/   # 多组训练结果
```

### DeepShip（新增）
```
results/
├── kfold_splits_deepship/           # K-Fold划分（单组）
├── kfold_cv_deepship/               # 训练结果（单组）
├── kfold_splits_multi_deepship/     # 多组划分
└── kfold_cv_deepship_multi/         # 多组训练结果
```

---

## 💻 完整使用流程

### 场景1：DeepShip快速对比（推荐）

```bash
# 一条命令，完整流程
python kfold_multi_train_deepship.py --all \
    --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/DeepShip_622 \
    --num-splits 3 \
    --gpus 4,5,6,7

# 查看对比结果
cat results/kfold_cv_deepship_multi/comparison_report.txt
```

### 场景2：对比ShipEar和DeepShip

```bash
# 训练ShipEar
python kfold_multi_train.py --all \
    --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622 \
    --num-splits 3 \
    --gpus 4,5,6,7

# 训练DeepShip
python kfold_multi_train_deepship.py --all \
    --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/DeepShip_622 \
    --num-splits 3 \
    --gpus 4,5,6,7

# 对比两个数据集的最佳性能
echo "=== ShipEar 最优结果 ===" && \
  grep "⭐" results/kfold_cv_shipsear_multi/comparison_report.txt

echo -e "\n=== DeepShip 最优结果 ===" && \
  grep "⭐" results/kfold_cv_deepship_multi/comparison_report.txt
```

### 场景3：自定义种子和参数

```bash
# 生成5组划分，使用自定义种子
python kfold_multi_split_generator.py \
    --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/DeepShip_622 \
    --output-dir results/kfold_splits_multi_deepship \
    --num-splits 5 \
    --seeds 42 100 200 300 400

# 然后训练
python kfold_deepship_integration.py --all \
    --gpus 4,5,6,7 \
    --splits-dir results/kfold_splits_multi_deepship/split_00_seed42
```

---

## 📊 输出示例

### 多划分对比报告

```
【各划分性能统计】

排名  划分名称                 平均精度    中位数      标准差      范围
─────────────────────────────────────────────────────────────────────
1     split_00_seed42          0.8945      0.8950      0.0156      [0.8600, 0.9200] ⭐
2     split_01_seed123         0.8812      0.8800      0.0189      [0.8400, 0.9100]
3     split_02_seed456         0.8756      0.8750      0.0213      [0.8300, 0.9050]

【推荐】
✓ 推荐使用: split_00_seed42
  平均精度最高，标准差最小
```

---

## ⚡ 快速命令参考

```bash
# 【完整流程一键执行】DeepShip
python kfold_multi_train_deepship.py --all \
    --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/DeepShip_622 \
    --num-splits 3 \
    --gpus 4,5,6,7

# 【只生成】多组划分
python kfold_multi_split_generator.py \
    --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/DeepShip_622 \
    --output-dir results/kfold_splits_multi_deepship \
    --num-splits 3

# 【只训练】所有split
python kfold_multi_train_deepship.py --train --gpus 4,5,6,7

# 【只对比】分析结果
python kfold_comparison_analyzer.py \
    --results-dir results/kfold_cv_deepship_multi

# 【查看报告】
cat results/kfold_cv_deepship_multi/comparison_report.txt

# 【查看划分对比】
cat results/kfold_splits_multi_deepship/splits_comparison.txt

# 【单个Fold测试】
python kfold_deepship_integration.py --fold 0 --gpus 4,5,6,7
```

---

## 📈 与ShipEar的对应关系

| 功能 | ShipEar | DeepShip |
|------|---------|----------|
| 单组K-Fold | `kfold_shipsear_integration.py` | `kfold_deepship_integration.py` |
| 多组K-Fold | `kfold_multi_train.py` | `kfold_multi_train_deepship.py` |
| K-Fold生成 | `kfold_multi_split_generator.py` (共用) | `kfold_multi_split_generator.py` (共用) |
| 对比分析 | `kfold_comparison_analyzer.py` (共用) | `kfold_comparison_analyzer.py` (共用) |

---

## 🎓 使用建议

### 👤 首次使用DeepShip

```bash
# 1. 快速测试（2个split）
python kfold_multi_train_deepship.py --all \
    --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/DeepShip_622 \
    --num-splits 2 \
    --gpus 4,5,6,7

# 2. 查看对比结果
cat results/kfold_cv_deepship_multi/comparison_report.txt

# 3. 使用最优划分
```

### 🔬 对比两个数据集

```bash
# 1. 生成两个数据集的K-Fold（可并行）
python kfold_multi_split_generator.py \
    --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622 \
    --output-dir results/kfold_splits_multi \
    --num-splits 3 &

python kfold_multi_split_generator.py \
    --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/DeepShip_622 \
    --output-dir results/kfold_splits_multi_deepship \
    --num-splits 3 &

wait

# 2. 训练两个数据集（可并行）
python kfold_multi_train.py --train --gpus 4,5,6,7 &
python kfold_multi_train_deepship.py --train --gpus 0,1,2,3 &
wait

# 3. 对比分析
python kfold_comparison_analyzer.py --results-dir results/kfold_cv_shipsear_multi
python kfold_comparison_analyzer.py --results-dir results/kfold_cv_deepship_multi
```

### 📚 论文实验

```bash
# 为两个数据集生成多组划分
python kfold_multi_train.py --all \
    --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622 \
    --num-splits 5 \
    --gpus 4,5,6,7

python kfold_multi_train_deepship.py --all \
    --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/DeepShip_622 \
    --num-splits 5 \
    --gpus 0,1,2,3

# 记录两个数据集的最优结果和对应的种子
```

---

## ✅ 检查清单

- [ ] 已确认DeepShip数据目录正确
- [ ] 已安装所有依赖
- [ ] 已决定生成多少组划分
- [ ] 已运行多组K-Fold生成脚本
- [ ] 已完成所有split的训练
- [ ] 已查看对比分析结果
- [ ] 已识别出最优划分
- [ ] 已保存结果到git
- [ ] 已在论文中记录使用的配置

---

## 🎉 总结

现在你拥有了**完整的ShipEar和DeepShip双数据集K-Fold系统**：

✅ **ShipEar K-Fold** - 完整的10折叠训练  
✅ **DeepShip K-Fold** - 完整的10折叠训练  
✅ **多组划分对比** - 两个数据集都支持  
✅ **自动性能分析** - 快速找到最优划分  
✅ **结果管理** - 清晰的目录结构  

**立即开始DeepShip训练：**

```bash
python kfold_multi_train_deepship.py --all \
    --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/DeepShip_622 \
    --num-splits 3 \
    --gpus 4,5,6,7
```

或对比两个数据集：

```bash
# ShipEar
python kfold_multi_train.py --all --num-splits 3 --gpus 4,5,6,7 &

# DeepShip（并行）
python kfold_multi_train_deepship.py --all --num-splits 3 --gpus 0,1,2,3 &

wait

echo "=== 结果对比 ===" && \
cat results/kfold_cv_shipsear_multi/comparison_report.txt && \
echo "" && \
cat results/kfold_cv_deepship_multi/comparison_report.txt
```

祝你找到最优的划分和模型！🚀
