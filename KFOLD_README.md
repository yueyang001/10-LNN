# K-Fold 十折叠交叉验证工具包

## 📦 包含文件

本工具包为LNN项目提供完整的十折叠交叉验证解决方案：

| 文件 | 功能 | 说明 |
|------|------|------|
| `kfold_cross_validation.py` | **主程序** | 生成K-Fold划分，保存到txt文档 |
| `kfold_data_loader.py` | **数据加载器** | 从保存的txt文件加载K-Fold数据 |
| `kfold_cv_training.py` | **训练集成** | 完整的K-Fold训练流程示例 |
| `KFOLD_USAGE_GUIDE.md` | **详细文档** | 完整使用说明和最佳实践 |

## 🎯 核心功能

✅ **平衡的数据划分**
- 使用StratifiedKFold确保每个折叠的类别分布一致
- 10个折叠，每个验证集占总数据的~10%

✅ **详细的划分结果保存**
- 每个折叠的完整样本列表（路径、标签、类别名）
- 类别分布统计
- 元数据（生成时间、随机种子、数据目录）

✅ **完全可复现**
- 固定的随机种子 (seed=42)
- 所有划分结果保存到txt，便于版本控制
- 使用相同参数可以完全复现结果

✅ **易于集成**
- 与现有训练脚本无缝集成
- 支持逐个训练Fold或批量训练
- 自动收集各Fold的训练结果

## 🚀 快速开始

### 第1步：修改数据目录

编辑 `kfold_cross_validation.py` 的最后部分：

```python
def main():
    data_dir = "./data"  # ← 改成你的数据路径
    # 例如: data_dir = "/mnt/data/ShipEar"
```

数据目录结构应该是：
```
data/
├── train/
│   ├── wav/
│   │   ├── A/     (或其他类别)
│   │   ├── B/
│   │   └── ...
│   ├── mel/
│   └── cqt/
├── validation/
│   └── ...
└── test/
    └── ...
```

### 第2步：生成K-Fold划分

```bash
python kfold_cross_validation.py
```

**输出示例：**
```
📊 开始扫描数据集...
✓ 扫描完成: train - 共 1000 个样本
  类别数: 5
  类别列表: ['A', 'B', 'C', 'D', 'E']

🔄 生成十折叠划分...
  Fold 0: train=900, val=100
    train标签分布: [180, 180, 180, 180, 180]
    val标签分布: [20, 20, 20, 20, 20]
  [... 9个Fold的信息 ...]

💾 保存划分结果...
✓ 已保存 Fold 0: results/kfold_splits/kfold_fold00.txt
[... 更多输出 ...]

✨ 完成! 所有文件已保存到: results/kfold_splits
```

### 第3步：查看划分结果

生成的txt文件结构：
```
results/kfold_splits/
├── kfold_summary.txt          # 10个Fold的汇总
├── kfold_fold00.txt           # Fold 0的详细信息（包含所有样本列表）
├── kfold_fold01.txt           # Fold 1的详细信息
├── ... (其他8个Fold)
└── kfold_indices.txt          # 索引形式的划分
```

查看汇总：
```bash
cat results/kfold_splits/kfold_summary.txt
```

查看Fold 0的所有样本：
```bash
cat results/kfold_splits/kfold_fold00.txt
```

### 第4步：在训练中使用

**简单方式：**
```python
from kfold_data_loader import KFoldDataLoader

# 加载Fold 0的数据
loader = KFoldDataLoader("results/kfold_splits/", fold_idx=0)
train_samples = loader.get_train_samples()
val_samples = loader.get_val_samples()

print(f"训练集: {len(train_samples)} 样本")
print(f"验证集: {len(val_samples)} 样本")

# train_samples 格式: [(wav_path, cat_idx, cat_name), ...]
for wav_path, cat_idx, cat_name in train_samples:
    # 你的处理逻辑
    pass
```

**完整示例：** 参考 `kfold_cv_training.py`

```bash
# 训练Fold 0
python kfold_cv_training.py --fold 0 --config configs/train_config.yaml

# 批量训练所有Fold（使用GPU 0,1,2,3）
python kfold_cv_training.py --all --gpus 0,1,2,3
```

## 📊 输出示例

### kfold_summary.txt
```
================================================================================
十折叠交叉验证 - 总体汇总
生成时间: 2025-05-02 10:00:00
随机种子: 42
数据目录: ./data
================================================================================

Fold 0:
  训练集: 900 样本 | 标签分布: [180, 180, 180, 180, 180]
  验证集: 100 样本 | 标签分布: [20, 20, 20, 20, 20]
  详细文件: kfold_fold00.txt

Fold 1:
  训练集: 900 样本 | 标签分布: [180, 180, 180, 180, 180]
  验证集: 100 样本 | 标签分布: [20, 20, 20, 20, 20]
  详细文件: kfold_fold01.txt

...
```

### kfold_fold00.txt (单个Fold详细信息)
```
================================================================================
十折叠交叉验证 - Fold 0
生成时间: 2025-05-02 10:00:00
随机种子: 42
================================================================================

【数据集统计】
训练集样本数: 900
验证集样本数: 100
训练集标签分布: [180, 180, 180, 180, 180]
验证集标签分布: [20, 20, 20, 20, 20]

------------ 【训练集 (Train Set)】 -----------
     1 | 0 | A                    | ./data/train/wav/A/sample_001.wav
     2 | 1 | B                    | ./data/train/wav/B/sample_101.wav
     3 | 2 | C                    | ./data/train/wav/C/sample_201.wav
     ...
   900 | 4 | E                    | ./data/train/wav/E/sample_999.wav

------------ 【验证集 (Validation Set)】 -----------
     1 | 0 | A                    | ./data/train/wav/A/sample_050.wav
     2 | 1 | B                    | ./data/train/wav/B/sample_150.wav
     ...
   100 | 4 | E                    | ./data/train/wav/E/sample_950.wav
```

## 🔄 复现性

### 确保完全复现

1. **使用相同的数据目录**
   ```python
   data_dir = "./data"  # 保持一致
   ```

2. **使用相同的随机种子**
   ```python
   validator = KFoldCrossValidator(..., seed=42)  # 保持一致
   ```

3. **保存划分结果到版本控制**
   ```bash
   git add results/kfold_splits/
   git commit -m "Add K-Fold splits (seed=42, 1000 samples)"
   ```

### 验证复现性

```bash
# 检查生成的txt文件是否相同
diff results/kfold_splits/kfold_fold00.txt backup_fold00.txt
# 如果没有输出，说明完全相同 ✓
```

## 📋 常见问题

### Q1: 我的数据结构不同怎么办？

编辑 `kfold_cross_validation.py` 的 `scan_dataset` 方法来适配你的数据结构。

### Q2: 如何更改Fold数量？

修改 `KFoldCrossValidator.__init__` 中的：
```python
self.n_splits = 10  # 改成你需要的数量，如5/3等
```

### Q3: 如何处理不平衡数据？

StratifiedKFold已经考虑了类别比例，会自动保持平衡。
查看输出的txt文件确认每个Fold的标签分布。

### Q4: 能否对train/val/test都进行K-Fold？

可以，多次运行：
```python
validator.scan_dataset("train")
validator.generate_folds(...)
validator.save_all_splits(suffix="_train")

validator.scan_dataset("validation")
validator.generate_folds(...)
validator.save_all_splits(suffix="_val")
```

## 📚 完整使用指南

详见 `KFOLD_USAGE_GUIDE.md`

## 🔗 相关链接

- sklearn.model_selection.StratifiedKFold: [官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
- K-Fold交叉验证原理: [参考资料](https://en.wikipedia.org/wiki/Cross-validation_(statistics))

## 💡 建议

1. **第一次运行时**
   - 检查生成的txt文件是否正确
   - 确认类别分布是否平衡
   - 验证样本路径是否正确

2. **使用前**
   - 备份划分结果到版本控制
   - 记录使用的数据集、样本数、种子等信息
   - 在论文/报告中引用

3. **批量训练时**
   - 先用一个Fold测试训练脚本
   - 确保路径和数据加载正确
   - 再运行所有Fold

## 📞 支持

有问题？检查：
1. 数据路径是否正确
2. 数据结构是否符合要求（train/val/test/wav/mel/cqt）
3. 是否有足够的磁盘空间
4. 查看详细的txt文件确认数据内容

---

**版本信息**
- 创建日期: 2025-05-02
- 最后更新: 2025-05-02
- 兼容Python: 3.7+
- 依赖: scikit-learn, numpy, PyYAML(可选)
