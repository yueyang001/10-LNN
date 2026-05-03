# K-Fold 十折叠交叉验证工具包 - 文件索引

## 📑 快速导航

| 需求 | 推荐文档 | 说明 |
|------|---------|------|
| ⚡ 快速上手 | `KFOLD_SUMMARY.md` | 30秒入门，包含所有基本信息 |
| 📖 详细学习 | `KFOLD_USAGE_GUIDE.md` | 完整参考手册 |
| 💻 代码查看 | `KFOLD_QUICK_REFERENCE.py` | 执行查看快速参考卡 |
| 🔧 编写脚本 | `kfold_data_loader.py` | API文档和示例 |

## 📚 文件清单

### 🎯 核心脚本（请按顺序使用）

```
1️⃣ kfold_cross_validation.py
   └─ 功能: 生成K-Fold划分，保存到txt
   └─ 用法: python kfold_cross_validation.py
   └─ 输出: results/kfold_splits/kfold_fold*.txt
   └─ 首次使用时修改: data_dir 变量

2️⃣ kfold_data_loader.py
   └─ 功能: 从txt文件加载K-Fold数据
   └─ 用法: from kfold_data_loader import KFoldDataLoader
   └─ 集成到你的训练脚本中

3️⃣ kfold_cv_training.py (可选)
   └─ 功能: 通用K-Fold训练示例
   └─ 用法: python kfold_cv_training.py --fold 0

4️⃣ kfold_shipsear_integration.py (ShipEar推荐)
   └─ 功能: ShipEar + K-Fold集成
   └─ 用法: python kfold_shipsear_integration.py --train-all --gpus 4,5,6,7
```

### 📖 文档文件

```
KFOLD_SUMMARY.md
  └─ 📌 总体汇总（推荐首先阅读）
  └─ 包含: 快速开始、文件说明、常见问题
  └─ 阅读时间: 5-10分钟

KFOLD_README.md
  └─ 📖 完整快速入门指南
  └─ 包含: 文件结构、详细步骤、使用示例
  └─ 阅读时间: 15-20分钟

KFOLD_USAGE_GUIDE.md
  └─ 🔍 详细参考手册（终极指南）
  └─ 包含: 所有细节、常见问题解决、最佳实践
  └─ 阅读时间: 30-45分钟

KFOLD_QUICK_REFERENCE.py
  └─ ⚡ 快速参考卡（可执行）
  └─ 用法: python KFOLD_QUICK_REFERENCE.py
  └─ 查看: 所有关键信息速查表

KFOLD_FILE_INDEX.md (本文件)
  └─ 🗂️ 文件导航和说明
```

## 🗂️ 目录结构

```
10-LNN/
│
├── 【K-Fold脚本】
│   ├── kfold_cross_validation.py       ← 第1步：生成划分
│   ├── kfold_data_loader.py            ← 第2步：加载数据
│   ├── kfold_cv_training.py            ← 第3步：训练示例
│   └── kfold_shipsear_integration.py   ← ShipEar推荐
│
├── 【K-Fold文档】
│   ├── KFOLD_SUMMARY.md                ← ⭐ 从这里开始
│   ├── KFOLD_README.md                 ← 快速入门
│   ├── KFOLD_USAGE_GUIDE.md            ← 详细参考
│   ├── KFOLD_QUICK_REFERENCE.py        ← 速查表
│   └── KFOLD_FILE_INDEX.md             ← 本文件
│
├── 【原有文件】
│   ├── experiment5_cross_validation.py
│   ├── experiment5_cross_validation_deepship.py
│   └── ...
│
└── 【输出目录（运行后自动生成）】
    └── results/
        └── kfold_splits/
            ├── kfold_summary.txt
            ├── kfold_fold00.txt
            ├── kfold_fold01.txt
            ├── ...
            ├── kfold_fold09.txt
            └── kfold_indices.txt
```

## 🚀 使用流程

### 流程1：首次设置（推荐所有用户）

```
1. 阅读文档
   ├─ 快速版: KFOLD_SUMMARY.md (5分钟)
   └─ 完整版: KFOLD_README.md (15分钟)

2. 修改配置
   └─ 编辑 kfold_cross_validation.py 中的 data_dir

3. 生成划分
   └─ python kfold_cross_validation.py

4. 验证结果
   └─ cat results/kfold_splits/kfold_summary.txt
   └─ cat results/kfold_splits/kfold_fold00.txt

5. 保存到git
   └─ git add results/kfold_splits/
   └─ git commit -m "Add K-Fold splits"
```

### 流程2：在代码中使用

```python
# 方式A：简单用法
from kfold_data_loader import KFoldDataLoader

loader = KFoldDataLoader("results/kfold_splits/", fold_idx=0)
train_samples = loader.get_train_samples()
val_samples = loader.get_val_samples()

# 方式B：批量加载所有Fold
from kfold_data_loader import load_kfold_splits

splits = load_kfold_splits("results/kfold_splits/")
for fold_idx, fold_data in splits.items():
    train = fold_data['train']
    val = fold_data['val']

# 方式C：ShipEar集成（推荐）
python kfold_shipsear_integration.py --train-all
```

### 流程3：ShipEar批量训练（推荐）

```bash
# 第1步: 生成K-Fold划分
python kfold_shipsear_integration.py --setup --data-dir ./data

# 第2步: 测试单个Fold
python kfold_shipsear_integration.py --train-fold 0 --gpus 4,5,6,7

# 第3步: 批量训练
python kfold_shipsear_integration.py --train-all --gpus 4,5,6,7

# 第4步: 查看结果
python kfold_shipsear_integration.py --results
```

## 🎓 学习路径

### 初级（快速上手）
```
目标: 在15分钟内生成K-Fold划分
├─ 阅读: KFOLD_SUMMARY.md 的"30秒快速开始"
├─ 修改: data_dir
├─ 运行: python kfold_cross_validation.py
└─ 验证: cat results/kfold_splits/kfold_summary.txt
```

### 中级（代码集成）
```
目标: 在训练代码中使用K-Fold数据
├─ 阅读: KFOLD_USAGE_GUIDE.md 的"在训练脚本中使用"
├─ 学习: kfold_data_loader.py 的API
├─ 参考: kfold_cv_training.py 的示例代码
└─ 实现: 在自己的训练脚本中集成
```

### 高级（完全理解）
```
目标: 理解所有细节，自定义和扩展
├─ 阅读: KFOLD_USAGE_GUIDE.md 的全部内容
├─ 研究: kfold_cross_validation.py 的实现细节
├─ 实验: 修改参数并观察结果
└─ 扩展: 根据需要修改和定制
```

## ✅ 检查清单

### 首次使用前
- [ ] 已阅读 KFOLD_SUMMARY.md
- [ ] 已检查数据目录结构正确
- [ ] 已修改 data_dir 变量

### 生成划分后
- [ ] 已运行 kfold_cross_validation.py
- [ ] 已验证生成的txt文件内容
- [ ] 已检查类别分布是否平衡

### 集成到训练前
- [ ] 已理解 KFoldDataLoader 的用法
- [ ] 已测试数据加载是否正确
- [ ] 已准备好修改训练脚本

### 生产就绪
- [ ] 已保存划分结果到git
- [ ] 已记录数据集和参数信息
- [ ] 已在项目文档中说明使用的K-Fold配置

## 🔗 相关文件对应关系

```
想要做... → 查看这个文件 → 然后使用这个脚本

了解工具功能
    ↓
    KFOLD_SUMMARY.md
    
想快速开始
    ↓
    KFOLD_README.md → kfold_cross_validation.py
    
想在代码中使用
    ↓
    KFOLD_USAGE_GUIDE.md → kfold_data_loader.py
    
想看完整参考
    ↓
    KFOLD_USAGE_GUIDE.md (+ 所有脚本注释)
    
想快速查找信息
    ↓
    KFOLD_QUICK_REFERENCE.py
    
想与ShipEar集成
    ↓
    KFOLD_SUMMARY.md或README.md → kfold_shipsear_integration.py
```

## 📋 常见任务对应文档

| 任务 | 对应文档章节 | 脚本 |
|------|------------|------|
| 修改数据目录 | KFOLD_USAGE_GUIDE.md - Q1 | kfold_cross_validation.py |
| 修改输出目录 | KFOLD_USAGE_GUIDE.md - Q2 | kfold_cross_validation.py |
| 修改随机种子 | KFOLD_USAGE_GUIDE.md - Q3 | kfold_cross_validation.py |
| 查看所有划分 | KFOLD_USAGE_GUIDE.md - Q4 | bash命令 |
| 加载单个Fold | KFOLD_USAGE_GUIDE.md - 方法A | kfold_data_loader.py |
| 完整训练流程 | KFOLD_USAGE_GUIDE.md - 方法B | kfold_cv_training.py |
| ShipEar集成 | KFOLD_SUMMARY.md - ShipEar | kfold_shipsear_integration.py |
| 保存到版本控制 | KFOLD_USAGE_GUIDE.md - 最佳实践 | git命令 |

## 🎯 推荐阅读顺序

### 对于不同背景的用户：

**想快速开始的人**
1. KFOLD_SUMMARY.md - 第1节（30秒）
2. KFOLD_SUMMARY.md - 第2节（快速开始）
3. 运行脚本，完成！

**需要完整理解的人**
1. KFOLD_SUMMARY.md - 快速总览
2. KFOLD_README.md - 完整流程
3. KFOLD_USAGE_GUIDE.md - 深入学习
4. 查看脚本注释 - 理解实现细节

**只想使用API的程序员**
1. KFOLD_SUMMARY.md - 第2节（快速开始）
2. KFOLD_USAGE_GUIDE.md - "在训练脚本中使用"
3. kfold_data_loader.py - 查看代码示例

**做论文研究的人**
1. KFOLD_README.md - 全部
2. KFOLD_USAGE_GUIDE.md - "复现性保证"和"最佳实践"
3. 保存到版本控制

## 💡 提示

- 📌 首次使用必读：KFOLD_SUMMARY.md
- 🔍 遇到问题先查：KFOLD_USAGE_GUIDE.md 的常见问题
- ⚡ 快速查找：执行 `python KFOLD_QUICK_REFERENCE.py`
- 📖 完整参考：始终可以回到 KFOLD_USAGE_GUIDE.md

## 🆘 需要帮助？

1. **问题**: 不知道从哪开始
   → 答案: 读 KFOLD_SUMMARY.md 的"30秒快速开始"

2. **问题**: 数据目录改哪儿
   → 答案: 查 KFOLD_USAGE_GUIDE.md 的 Q1

3. **问题**: 怎么在代码中使用
   → 答案: 看 KFOLD_USAGE_GUIDE.md 的"在训练脚本中使用"

4. **问题**: 与ShipEar怎么集成
   → 答案: 看 KFOLD_SUMMARY.md 的"与ShipEar集成"

5. **问题**: 结果能否复现
   → 答案: 查 KFOLD_USAGE_GUIDE.md 的"复现性保证"

---

**版本**: 1.0  
**创建**: 2025-05-02  
**最后更新**: 2025-05-02

所有脚本已准备就绪，文档完整详细。祝使用愉快！ 🎉
