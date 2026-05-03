# K-Fold 十折叠交叉验证使用指南

## 📋 概述

这套工具提供了完整的十折叠交叉验证解决方案，用于:
- 生成平衡的K-Fold数据划分
- 保存详细的划分结果到txt文档
- 支持完全复现 (使用固定随机种子)
- 方便地集成到现有训练脚本

## 📂 文件结构

```
10-LNN/
├── kfold_cross_validation.py      # 主程序：生成K-Fold划分
├── kfold_data_loader.py           # 辅助模块：加载划分结果
├── kfold_cv_training.py           # 集成脚本：完整的K-Fold训练示例
├── results/
│   └── kfold_splits/              # 输出目录
│       ├── kfold_summary.txt      # 总体汇总
│       ├── kfold_fold00.txt       # Fold 0详细信息
│       ├── kfold_fold01.txt       # Fold 1详细信息
│       ├── ...
│       └── kfold_indices.txt      # 索引形式
```

## 🚀 快速开始

### 1️⃣ 生成K-Fold划分

修改数据目录，然后运行:

```bash
python kfold_cross_validation.py
```

此步骤会:
- 扫描 `./data/train` 目录
- 使用StratifiedKFold生成10个平衡的折叠
- 保存所有划分结果到 `results/kfold_splits/`

**输出示例:**
```
📊 开始扫描数据集...
✓ 扫描完成: train - 共 1000 个样本
  类别数: 5
  类别列表: ['A', 'B', 'C', 'D', 'E']

🔄 生成十折叠划分...
  Fold 0: train=900, val=100
    train标签分布: [180, 180, 180, 180, 180]
    val标签分布: [20, 20, 20, 20, 20]
  ...

💾 保存划分结果...
✓ 已保存 Fold 0: results/kfold_splits/kfold_fold00.txt
✓ 已保存 Fold 1: results/kfold_splits/kfold_fold01.txt
...
```

### 2️⃣ 查看划分结果

打开生成的txt文件查看详细信息：

**kfold_summary.txt** - 总体汇总:
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
...
```

**kfold_fold00.txt** - 单个折叠详细信息:
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
     ...
```

### 3️⃣ 在训练脚本中使用

#### 方法A: 使用KFoldDataLoader加载划分

```python
from kfold_data_loader import KFoldDataLoader

# 加载第0折
loader = KFoldDataLoader("results/kfold_splits/", fold_idx=0)
train_samples = loader.get_train_samples()
val_samples = loader.get_val_samples()

print(f"训练集: {len(train_samples)} 样本")
print(f"验证集: {len(val_samples)} 样本")

# train_samples 格式: [(wav_path, cat_idx, cat_name), ...]
for wav_path, cat_idx, cat_name in train_samples[:3]:
    print(f"  {wav_path}")
```

#### 方法B: 完整的K-Fold训练流程

参考 `kfold_cv_training.py` 了解完整示例。

## 🔄 复现性保证

所有划分结果都完全可复现，因为:

1. **固定的随机种子**
   ```python
   validator = KFoldCrossValidator(
       data_dir="./data",
       seed=42  # ⚠️ 重要：保持一致
   )
   ```

2. **详细的元数据**
   - 每个txt文件都记录生成时间、随机种子、数据目录
   - 完整列出所有样本路径

3. **版本控制**
   ```bash
   # 把所有txt文件添加到git
   git add results/kfold_splits/*.txt
   git commit -m "Add K-Fold split results"
   ```

**复现步骤:**
```bash
# 克隆代码，修改数据路径
git clone ...
cd 10-LNN

# 使用相同参数重新生成
python kfold_cross_validation.py

# 生成的txt文件应该与之前完全相同
diff results/kfold_splits/kfold_fold00.txt <other_fold00.txt>
```

## 📊 K-Fold 统计示例

以5个类别、1000个样本为例：

```
Fold 0: train=900, val=100
  train类别分布: [180, 180, 180, 180, 180]  ✓ 完全平衡
  val类别分布:   [20,  20,  20,  20,  20]   ✓ 完全平衡

Fold 1: train=900, val=100
  train类别分布: [180, 180, 180, 180, 180]  ✓ 完全平衡
  val类别分布:   [20,  20,  20,  20,  20]   ✓ 完全平衡
...
```

## 🛠️ 常见问题

### Q1: 如何修改数据目录?

编辑 `kfold_cross_validation.py` 的 `main()` 函数:

```python
def main():
    # 修改这一行
    data_dir = "/path/to/your/data"  # ← 改这里
    
    validator = KFoldCrossValidator(
        data_dir=data_dir,
        output_dir="results/kfold_splits",
        seed=42
    )
```

### Q2: 如何修改输出目录?

```python
validator = KFoldCrossValidator(
    data_dir="./data",
    output_dir="my_custom_output_dir",  # ← 改这里
    seed=42
)
```

### Q3: 如何修改随机种子?

```python
validator = KFoldCrossValidator(
    data_dir="./data",
    output_dir="results/kfold_splits",
    seed=123  # ← 改这里，不同的seed会生成不同的划分
)
```

### Q4: 如何查看所有划分信息?

```bash
# 查看汇总信息
cat results/kfold_splits/kfold_summary.txt

# 查看单个折叠
cat results/kfold_splits/kfold_fold00.txt

# 查看索引形式
cat results/kfold_splits/kfold_indices.txt
```

## 📝 脚本集成示例

### 与现有训练脚本集成

```python
import yaml
from kfold_data_loader import KFoldDataLoader

def train_with_kfold(fold_idx, config_path):
    """训练单个K-Fold"""
    
    # 加载K-Fold划分
    loader = KFoldDataLoader("results/kfold_splits/", fold_idx)
    train_samples = loader.get_train_samples()
    val_samples = loader.get_val_samples()
    
    # 保存到config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    config["dataset"]["fold"] = fold_idx
    config["dataset"]["train_samples"] = train_samples
    config["dataset"]["val_samples"] = val_samples
    
    # 启动训练
    train_model(config)

# 批量训练所有K-Fold
for fold_idx in range(10):
    train_with_kfold(fold_idx, "configs/train_config.yaml")
```

## 📌 最佳实践

1. **保存划分结果到版本控制**
   ```bash
   git add results/kfold_splits/
   git commit -m "Add K-Fold splits (seed=42)"
   ```

2. **记录超参数和设置**
   ```
   # 在README中记录
   - K-Fold种子: 42
   - 数据集: ShipEar (1000 samples, 5 classes)
   - 划分文件: results/kfold_splits/
   ```

3. **验证样本平衡**
   ```python
   # 检查每个fold的类别分布
   cat results/kfold_splits/kfold_summary.txt
   # 确认所有train/val的类别分布都相同
   ```

## 🔗 相关文件

- `kfold_cross_validation.py` - K-Fold生成主程序
- `kfold_data_loader.py` - 数据加载器
- `kfold_cv_training.py` - 完整集成示例

## 📧 使用建议

- 第一次运行时，检查输出的txt文件确保数据正确
- 把txt文件放入版本控制确保复现性
- 在论文/报告中引用使用的随机种子和划分文件
