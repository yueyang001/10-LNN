"""
快速参考卡 - K-Fold十折叠交叉验证

════════════════════════════════════════════════════════════════════════════════
📦 生成的文件列表
════════════════════════════════════════════════════════════════════════════════

主程序脚本:
  ✓ kfold_cross_validation.py       - 生成K-Fold划分
  ✓ kfold_data_loader.py            - 数据加载器
  ✓ kfold_cv_training.py            - 通用训练示例
  ✓ kfold_shipsear_integration.py   - ShipEar集成脚本

说明文档:
  ✓ KFOLD_README.md                 - 快速入门指南
  ✓ KFOLD_USAGE_GUIDE.md            - 详细使用说明

════════════════════════════════════════════════════════════════════════════════
🚀 快速开始（3个命令）
════════════════════════════════════════════════════════════════════════════════

# 第1步：修改kfold_cross_validation.py中的数据目录
data_dir = "./data"  # 改成你的数据路径

# 第2步：生成K-Fold划分
python kfold_cross_validation.py

# 第3步：查看结果
cat results/kfold_splits/kfold_summary.txt

════════════════════════════════════════════════════════════════════════════════
📂 输出文件结构
════════════════════════════════════════════════════════════════════════════════

results/kfold_splits/
├── kfold_summary.txt          # 所有Fold的汇总信息
├── kfold_fold00.txt           # Fold 0 的详细信息（包含全部样本列表）
├── kfold_fold01.txt           # Fold 1 的详细信息
├── kfold_fold02.txt           # Fold 2 的详细信息
├── ...
├── kfold_fold09.txt           # Fold 9 的详细信息
└── kfold_indices.txt          # 索引形式的划分结果

每个txt文件包含:
  - 数据统计（训练集/验证集样本数、类别分布）
  - 完整的样本列表（路径、标签、类别名）
  - 元数据（生成时间、随机种子、数据目录）

════════════════════════════════════════════════════════════════════════════════
💻 Python代码使用示例
════════════════════════════════════════════════════════════════════════════════

# 方式1：加载单个Fold
from kfold_data_loader import KFoldDataLoader

loader = KFoldDataLoader("results/kfold_splits/", fold_idx=0)
train_samples = loader.get_train_samples()   # [(path, idx, name), ...]
val_samples = loader.get_val_samples()       # [(path, idx, name), ...]

print(f"训练集: {len(train_samples)} 样本")
print(f"验证集: {len(val_samples)} 样本")

# 方式2：加载所有Fold
from kfold_data_loader import load_kfold_splits

splits = load_kfold_splits("results/kfold_splits/")
for fold_idx, fold_data in splits.items():
    print(f"Fold {fold_idx}: train={len(fold_data['train'])}, val={len(fold_data['val'])}")

════════════════════════════════════════════════════════════════════════════════
🔄 ShipEar集成（推荐方式）
════════════════════════════════════════════════════════════════════════════════

第1步：生成K-Fold划分
  python kfold_shipsear_integration.py --setup --data-dir ./data

第2步：训练单个Fold（测试）
  python kfold_shipsear_integration.py --train-fold 0 --gpus 4,5,6,7

第3步：批量训练所有Fold
  python kfold_shipsear_integration.py --train-all --gpus 4,5,6,7

第4步：查看结果
  python kfold_shipsear_integration.py --results
  cat results/kfold_cv_shipsear/training_summary.txt

════════════════════════════════════════════════════════════════════════════════
📊 数据统计示例
════════════════════════════════════════════════════════════════════════════════

假设数据: 1000个样本 × 5个类别

Fold 0:
  ✓ 训练集: 900 样本 | 类别分布: [180, 180, 180, 180, 180]  完全平衡
  ✓ 验证集: 100 样本 | 类别分布: [20, 20, 20, 20, 20]        完全平衡

Fold 1:
  ✓ 训练集: 900 样本 | 类别分布: [180, 180, 180, 180, 180]  完全平衡
  ✓ 验证集: 100 样本 | 类别分布: [20, 20, 20, 20, 20]        完全平衡

... (类似的8个Fold)

════════════════════════════════════════════════════════════════════════════════
⚙️ 关键配置参数
════════════════════════════════════════════════════════════════════════════════

kfold_cross_validation.py:
  data_dir = "./data"                    # 修改为你的数据路径
  output_dir = "results/kfold_splits"    # 输出目录
  seed = 42                              # 随机种子（保证复现性）
  n_splits = 10                          # Fold数量

数据目录结构要求:
  data/
  ├── train/
  │   ├── wav/     (必需)
  │   ├── mel/     (可选，根据项目)
  │   └── cqt/     (可选，根据项目)
  ├── validation/
  └── test/

════════════════════════════════════════════════════════════════════════════════
🔒 确保复现性的3个步骤
════════════════════════════════════════════════════════════════════════════════

1. 使用相同的数据目录
   data_dir = "./data"  # ← 保持一致

2. 使用相同的随机种子
   seed = 42  # ← 保持一致

3. 保存划分结果到版本控制
   git add results/kfold_splits/
   git commit -m "Add K-Fold splits (seed=42, ShipEar 1000 samples)"

验证复现性:
   diff results/kfold_splits/kfold_fold00.txt backup_fold00.txt
   # 如果没有输出，说明完全相同 ✓

════════════════════════════════════════════════════════════════════════════════
📝 在论文/报告中如何引用
════════════════════════════════════════════════════════════════════════════════

方法部分:
  "We use 10-fold stratified cross-validation to evaluate our model.
   The data split is generated using scikit-learn's StratifiedKFold with seed=42.
   Each fold maintains the original class distribution."

表格或图表说明:
  "Table X shows the 10-fold cross-validation results. The data splitting
   configuration is reproducible and stored in results/kfold_splits/."

复现性说明:
  "All K-fold splits and their detailed sample lists are available at
   results/kfold_splits/kfold_fold*.txt for complete reproducibility."

════════════════════════════════════════════════════════════════════════════════
🐛 常见问题排查
════════════════════════════════════════════════════════════════════════════════

问题1：找不到数据
  → 检查 data_dir 是否正确
  → 确保目录结构: train/wav/, train/mel/, train/cqt/
  → 检查wav文件夹内是否有类别子文件夹

问题2：无法生成txt文件
  → 检查 results/kfold_splits/ 目录是否存在且可写
  → 检查磁盘空间是否充足

问题3：结果不复现
  → 检查随机种子是否一致 (seed=42)
  → 确保使用相同的数据目录
  → 验证scikit-learn版本一致

问题4：类别分布不平衡
  → StratifiedKFold会自动保持平衡
  → 查看输出txt文件中的"【数据集统计】"部分
  → 如果仍有问题，检查原始数据是否平衡

════════════════════════════════════════════════════════════════════════════════
📚 相关资源
════════════════════════════════════════════════════════════════════════════════

完整文档: 见 KFOLD_USAGE_GUIDE.md
详细代码: 见 kfold_cross_validation.py 中的注释

K-Fold交叉验证理论:
  https://en.wikipedia.org/wiki/Cross-validation_(statistics)

scikit-learn StratifiedKFold:
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html

════════════════════════════════════════════════════════════════════════════════
✅ 检查清单
════════════════════════════════════════════════════════════════════════════════

准备阶段:
  □ 确认数据目录路径正确
  □ 确保数据结构符合要求 (train/validation/test, wav/mel/cqt)
  □ 验证有足够的磁盘空间

生成划分:
  □ 修改 data_dir 变量
  □ 运行 kfold_cross_validation.py
  □ 检查生成的 txt 文件
  □ 验证类别分布是否平衡

集成到训练:
  □ 使用 KFoldDataLoader 加载数据
  □ 或使用 kfold_shipsear_integration.py
  □ 验证样本路径正确
  □ 测试单个Fold的训练

版本控制:
  □ git add results/kfold_splits/
  □ git commit -m "Add K-Fold splits"
  □ 记录数据集信息和参数

════════════════════════════════════════════════════════════════════════════════

生成时间: 2025-05-02
版本: 1.0
兼容Python: 3.7+
依赖: scikit-learn, numpy
"""

if __name__ == "__main__":
    # 打印本文件内容
    print(__doc__)
