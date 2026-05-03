# K-Fold十折叠交叉验证工具包 - 使用汇总

## 📦 创建的文件

已为您的LNN项目创建了完整的K-Fold十折叠交叉验证工具包：

### 核心脚本（4个）
1. **`kfold_cross_validation.py`** - 主程序
   - 扫描数据目录
   - 生成平衡的10-Fold划分
   - 保存详细的txt划分结果
   - **关键功能**：StratifiedKFold平衡划分、详细元数据记录

2. **`kfold_data_loader.py`** - 数据加载器
   - 从txt文件加载K-Fold划分
   - 支持Python集成
   - **关键功能**：解析txt文件、返回结构化数据

3. **`kfold_cv_training.py`** - 通用训练示例
   - 演示如何集成K-Fold
   - 支持单个Fold或批量训练
   - 自动收集结果统计
   - **关键功能**：训练流程示例、结果汇总

4. **`kfold_shipsear_integration.py`** - ShipEar集成脚本
   - 与ShipEar训练系统直接集成
   - 命令行工具
   - **关键功能**：一键生成、训练、统计

### 文档（3个）
1. **`KFOLD_README.md`** - 快速入门指南（5分钟上手）
2. **`KFOLD_USAGE_GUIDE.md`** - 详细使用说明（完整参考）
3. **`KFOLD_QUICK_REFERENCE.py`** - 快速参考卡（可执行查看）

## 🚀 30秒快速开始

### 第1步：修改数据目录
编辑 `kfold_cross_validation.py` 的 `main()` 函数：
```python
def main():
    data_dir = "/path/to/your/data"  # ← 改成你的数据路径
    # 例如: "./data" 或 "/mnt/data/ShipEar"
```

### 第2步：运行生成脚本
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
  ...

💾 保存划分结果...
✓ 已保存 Fold 0: results/kfold_splits/kfold_fold00.txt
[... 更多输出 ...]

✨ 完成! 所有文件已保存到: results/kfold_splits
```

### 第3步：查看结果
```bash
# 查看汇总信息
cat results/kfold_splits/kfold_summary.txt

# 查看Fold 0的详细信息（包含全部样本列表）
cat results/kfold_splits/kfold_fold00.txt
```

## 📂 输出文件结构

```
results/kfold_splits/
├── kfold_summary.txt          # 所有10个Fold的汇总
│   └── 包含: 每个Fold的样本数、类别分布
│
├── kfold_fold00.txt           # Fold 0详细信息
├── kfold_fold01.txt           # Fold 1详细信息
├── ...
└── kfold_fold09.txt           # Fold 9详细信息
    └── 每个文件包含:
        - 数据集统计
        - 完整的训练集样本列表（格式：idx | 类别id | 类别名 | 路径）
        - 完整的验证集样本列表
        - 元数据（生成时间、随机种子等）
```

## 💡 关键特性

### ✅ 平衡的数据划分
- 使用 `StratifiedKFold` 确保每个折叠的类别分布完全一致
- 自动处理不平衡数据

### ✅ 详细的划分结果
```
【数据集统计】
训练集样本数: 900
验证集样本数: 100
训练集标签分布: [180, 180, 180, 180, 180]
验证集标签分布: [20, 20, 20, 20, 20]

【训练集 (Train Set)】
     1 | 0 | A                    | ./data/train/wav/A/sample_001.wav
     2 | 1 | B                    | ./data/train/wav/B/sample_101.wav
     ...
   900 | 4 | E                    | ./data/train/wav/E/sample_999.wav

【验证集 (Validation Set)】
     1 | 0 | A                    | ./data/train/wav/A/sample_050.wav
     ...
```

### ✅ 完全可复现
- 固定的随机种子 (seed=42)
- 所有划分结果保存到txt
- 版本控制友好

### ✅ 易于集成
```python
from kfold_data_loader import KFoldDataLoader

# 加载Fold 0
loader = KFoldDataLoader("results/kfold_splits/", fold_idx=0)
train_samples = loader.get_train_samples()  # [(path, idx, name), ...]
val_samples = loader.get_val_samples()      # [(path, idx, name), ...]

for wav_path, cat_idx, cat_name in train_samples:
    # 你的处理逻辑
    pass
```

## 🔗 与ShipEar集成（推荐）

对于ShipEar项目，使用集成脚本更方便：

```bash
# 第1步：生成K-Fold划分
python kfold_shipsear_integration.py --setup --data-dir ./data

# 第2步：测试单个Fold（可选）
python kfold_shipsear_integration.py --train-fold 0 --gpus 4,5,6,7

# 第3步：批量训练所有Fold
python kfold_shipsear_integration.py --train-all --gpus 4,5,6,7

# 第4步：查看结果
python kfold_shipsear_integration.py --results
```

## 📊 结果示例

### kfold_summary.txt（汇总）
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

... (其他8个Fold类似)
```

### kfold_fold00.txt（单个Fold详细）
完整列出所有900个训练样本和100个验证样本的路径、标签等信息。

## 🔒 确保复现性

```bash
# 1. 保存划分结果到版本控制
git add results/kfold_splits/
git commit -m "Add K-Fold splits (seed=42, ShipEar 1000 samples)"

# 2. 记录配置信息
# 在项目README中添加:
# - K-Fold种子: 42
# - 数据集: ShipEar
# - 样本数: 1000
# - 类别数: 5
# - 划分文件: results/kfold_splits/

# 3. 验证复现性
diff results/kfold_splits/kfold_fold00.txt backup_fold00.txt
# 如果没有输出，说明完全相同 ✓
```

## 📖 完整文档

- **快速入门**: `KFOLD_README.md` (5-10分钟)
- **详细说明**: `KFOLD_USAGE_GUIDE.md` (完整参考)
- **快速查看**: `python KFOLD_QUICK_REFERENCE.py` (可执行)

## 🎯 典型用途

### 用途1：论文实验
```
1. 生成K-Fold划分
2. 把txt文件加入版本控制
3. 在论文中引用"10-fold cross-validation with seed=42"
4. 其他研究者可以完全复现结果
```

### 用途2：模型评估
```
1. 生成K-Fold划分
2. 用kfold_cv_training.py运行所有10个Fold
3. 收集每个Fold的精度
4. 计算平均精度和标准差
5. 得出更鲁棒的模型性能估计
```

### 用途3：超参数调优
```
1. 生成K-Fold划分（固定）
2. 对每个Fold运行网格搜索
3. 比较不同超参数的跨折叠平均性能
4. 选择最优超参数
```

## ⚙️ 数据目录要求

```
data/
├── train/
│   ├── wav/
│   │   ├── A/
│   │   │   ├── sample_001.wav
│   │   │   ├── sample_002.wav
│   │   │   └── ...
│   │   ├── B/
│   │   ├── C/
│   │   ├── D/
│   │   └── E/
│   ├── mel/
│   │   ├── A/
│   │   │   ├── sample_001@mel.png
│   │   │   └── ...
│   │   └── ...
│   └── cqt/
│       ├── A/
│       │   ├── sample_001@cqt.png
│       │   └── ...
│       └── ...
├── validation/  (结构同train)
└── test/        (结构同train)
```

## ❓ 常见问题

**Q: 如何修改Fold数量？**
A: 编辑 `kfold_cross_validation.py`，修改 `self.n_splits = 10` 为你需要的数量。

**Q: 如何验证划分是否正确？**
A: 查看生成的txt文件，检查标签分布是否平衡。

**Q: 能否对不同的数据集运行多次？**
A: 可以，使用不同的输出目录：`validator.save_all_splits(suffix="_dataset1")`

**Q: 如何与现有训练脚本集成？**
A: 使用 `KFoldDataLoader` 加载数据，参考 `kfold_cv_training.py` 中的示例。

## 📝 最后检查清单

- [ ] 修改了 `data_dir` 指向正确的数据路径
- [ ] 验证了数据目录结构（train/validation/test, wav/mel/cqt）
- [ ] 运行了 `kfold_cross_validation.py` 生成划分
- [ ] 查看了生成的 txt 文件确保内容正确
- [ ] 验证了类别分布是否平衡
- [ ] （可选）已保存划分结果到git
- [ ] 准备好在训练脚本中集成 K-Fold

---

**创建日期**: 2025-05-02  
**兼容版本**: Python 3.7+  
**依赖**: scikit-learn, numpy, PyYAML(可选)

有任何问题，请查看 `KFOLD_USAGE_GUIDE.md` 中的"常见问题"部分。
