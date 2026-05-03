# 🚀 K-Fold 十折叠交叉验证 - 完整使用指南

## 快速开始（选择一个方法）

### 方法1️⃣：最简单 - 使用Python交互式指南（推荐首次）

```bash
python kfold_quick_start.py
```

这会：
- ✓ 检查环境和依赖
- ✓ 引导你配置数据目录
- ✓ 逐步生成K-Fold划分
- ✓ 验证划分结果
- ✓ 帮助你测试和运行训练

### 方法2️⃣：最快 - 使用Bash脚本

```bash
bash kfold_run.sh
```

交互式菜单选择要执行的操作。

### 方法3️⃣：最直接 - 使用命令行

逐条执行命令（见下方）。

---

## 完整的命令行流程

### 第1步：生成K-Fold划分（第一次只需做一次）

编辑 `kfold_cross_validation.py`，修改数据目录：

```python
def main():
    data_dir = "/media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622"  # ← 改这里
```

然后运行：

```bash
python kfold_cross_validation.py
```

**预期输出:**
```
📊 开始扫描数据集...
✓ 扫描完成: train - 共 1000 个样本
  类别数: 5
  类别列表: ['A', 'B', 'C', 'D', 'E']

🔄 生成十折叠划分...
  Fold 0: train=900, val=100
    train标签分布: [180, 180, 180, 180, 180]
    val标签分布: [20, 20, 20, 20, 20]
  [... 更多Fold ...]

💾 保存划分结果...
✓ 已保存 Fold 0: results/kfold_splits/kfold_fold00.txt
[... 更多文件 ...]

✨ 完成! 所有文件已保存到: results/kfold_splits
```

### 第2步：验证划分结果

```bash
# 查看汇总信息
cat results/kfold_splits/kfold_summary.txt

# 查看Fold 0的详细信息
cat results/kfold_splits/kfold_fold00.txt

# 统计样本数
wc -l results/kfold_splits/kfold_fold*.txt
```

### 第3步：测试单个Fold（可选但推荐）

```bash
# 测试Fold 0的训练
python kfold_shipsear_integration.py \
    --train-fold 0 \
    --gpus 4,5,6,7 \
    --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622
```

**观察输出，确保:**
- ✓ 数据加载正确
- ✓ 训练开始
- ✓ 没有路径或模块错误
- ✓ GPU正常使用

如果Fold 0成功，说明整个流程没问题。

### 第4步：批量训练所有Fold

```bash
# 顺序训练（推荐）
python kfold_shipsear_integration.py \
    --train-all \
    --gpus 4,5,6,7 \
    --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622
```

**或者并行训练所有Fold（更快）:**

```bash
# 在后台启动所有10个Fold的训练
for fold_idx in {0..9}; do
    echo "Starting Fold $fold_idx..."
    python kfold_shipsear_integration.py \
        --train-fold $fold_idx \
        --gpus 4,5,6,7 \
        --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622 &
done

# 等待所有训练完成
wait

echo "所有Fold训练完成!"
```

**监控训练进度:**

```bash
# 查看GPU使用情况
watch -n 1 nvidia-smi

# 或者检查生成的日志
ls -lth checkpoints/cv_shipsear/fold_*/training.log
```

### 第5步：查看训练结果

```bash
# 查看结果摘要
python kfold_shipsear_integration.py --results

# 或查看详细的CSV文件
cat results/kfold_cv_shipsear/kfold_shipsear_results.csv

# 或查看完整报告
cat results/kfold_cv_shipsear/training_summary.txt
```

---

## 在Python代码中使用K-Fold数据

### 简单用法：加载单个Fold

```python
from kfold_data_loader import KFoldDataLoader

# 加载Fold 0
loader = KFoldDataLoader("results/kfold_splits/", fold_idx=0)
train_samples = loader.get_train_samples()
val_samples = loader.get_val_samples()

print(f"训练集: {len(train_samples)} 样本")
print(f"验证集: {len(val_samples)} 样本")

# train_samples 格式: [(wav_path, cat_idx, cat_name), ...]
for wav_path, cat_idx, cat_name in train_samples[:3]:
    print(f"  {cat_name}: {wav_path}")
```

### 批量用法：加载所有Fold

```python
from kfold_data_loader import load_kfold_splits

# 加载所有10个Fold
splits = load_kfold_splits("results/kfold_splits/")

for fold_idx, fold_data in splits.items():
    train = fold_data['train']
    val = fold_data['val']
    print(f"Fold {fold_idx}: train={len(train)}, val={len(val)}")
```

### 集成到你的训练脚本

```python
import yaml
from kfold_data_loader import KFoldDataLoader

def train_with_kfold():
    # 加载配置
    with open("configs/train_distillation_shipsear.yaml") as f:
        config = yaml.safe_load(f)
    
    # 对每个Fold训练
    for fold_idx in range(10):
        # 加载该Fold的数据
        loader = KFoldDataLoader("results/kfold_splits/", fold_idx)
        
        # 更新配置
        config['dataset']['fold'] = fold_idx
        config['dataset']['train_samples'] = loader.get_train_samples()
        config['dataset']['val_samples'] = loader.get_val_samples()
        
        # 设置输出目录
        config['save']['save_dir'] = f"checkpoints/cv_shipsear/fold_{fold_idx:02d}"
        
        # 运行训练（调用你的训练函数）
        train_model(config)

if __name__ == "__main__":
    train_with_kfold()
```

---

## 常用命令速查表

```bash
# 【生成和验证】
python kfold_cross_validation.py                    # 生成K-Fold划分
cat results/kfold_splits/kfold_summary.txt         # 查看汇总
cat results/kfold_splits/kfold_fold00.txt          # 查看Fold 0详细信息

# 【运行训练】
python kfold_shipsear_integration.py --train-fold 0 --gpus 4,5,6,7     # 训练Fold 0
python kfold_shipsear_integration.py --train-all --gpus 4,5,6,7        # 训练所有Fold

# 【查看结果】
python kfold_shipsear_integration.py --results       # 查看结果摘要
cat results/kfold_cv_shipsear/kfold_shipsear_results.csv   # 详细CSV

# 【监控】
nvidia-smi                                           # 查看GPU使用
watch -n 1 nvidia-smi                               # 实时监控GPU

# 【快速启动】
python kfold_quick_start.py                         # Python交互式指南
bash kfold_run.sh                                   # Bash交互式菜单
```

---

## 数据目录结构

确保数据目录结构正确：

```
/media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622/
├── train/
│   ├── wav/
│   │   ├── A/
│   │   │   ├── sample_001.wav
│   │   │   ├── sample_002.wav
│   │   │   └── ... (所有A类的wav文件)
│   │   ├── B/      (同样结构)
│   │   ├── C/
│   │   ├── D/
│   │   └── E/
│   ├── mel/
│   │   ├── A/
│   │   │   ├── sample_001@mel.png
│   │   │   └── ...
│   │   ├── B/, C/, D/, E/  (同样结构)
│   └── cqt/
│       ├── A/, B/, C/, D/, E/  (同样结构)
├── validation/  (结构同train)
└── test/        (结构同train)
```

---

## 输出文件说明

生成和训练后会产生以下文件：

```
results/
├── kfold_splits/
│   ├── kfold_summary.txt           # 所有Fold的汇总
│   ├── kfold_fold00.txt            # Fold 0详细信息
│   ├── kfold_fold01.txt
│   ├── ...
│   ├── kfold_fold09.txt
│   └── kfold_indices.txt           # 索引形式
│
└── kfold_cv_shipsear/              # 训练结果（运行训练后生成）
    ├── kfold_shipsear_results.csv  # 结果摘要
    ├── training_summary.txt        # 完整报告
    └── fold_0*/
        fold_1*/
        ... (每个Fold的检查点)
```

---

## 常见问题解决

### Q1: 无法找到数据

```bash
# 检查数据路径
ls /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622/train/wav/

# 修改kfold_cross_validation.py中的data_dir
data_dir = "/你/的/实际/数据/路径"
```

### Q2: GPU内存不足

```python
# 修改config文件中的batch_size
training:
  batch_size: 8  # 从16改小到8
```

### Q3: Fold训练失败

```bash
# 检查单个Fold 0是否成功
python kfold_shipsear_integration.py --train-fold 0 --gpus 4,5,6,7

# 查看详细错误
tail -f checkpoints/cv_shipsear/fold_00/training.log
```

### Q4: 如何删除重新开始

```bash
# 删除K-Fold划分结果
rm -rf results/kfold_splits/

# 删除训练结果
rm -rf results/kfold_cv_shipsear/
rm -rf checkpoints/cv_shipsear/

# 重新开始
python kfold_cross_validation.py
```

---

## 最佳实践

### ✅ 检查清单

- [ ] 数据目录结构正确
- [ ] Python依赖已安装
- [ ] K-Fold划分已生成
- [ ] 划分结果已验证（类别分布平衡）
- [ ] Fold 0测试成功
- [ ] 已保存到git（可选但推荐）

### ✅ 保存结果到版本控制

```bash
# 保存K-Fold划分结果
git add results/kfold_splits/
git add results/kfold_cv_shipsear/
git commit -m "Add K-Fold cross-validation (ShipEar, seed=42, 10 folds)"

# 在项目文档中记录
# - K-Fold种子: 42
# - 数据集: ShipEar 1000 samples
# - 划分文件: results/kfold_splits/
```

### ✅ 监控长时间训练

```bash
# 后台运行并保存日志
python kfold_shipsear_integration.py --train-all > train.log 2>&1 &

# 监控进度
tail -f train.log

# 后来查看
cat train.log | grep "Fold\|acc\|success"
```

---

## 文档索引

| 文件 | 说明 |
|------|------|
| `KFOLD_SUMMARY.md` | 总体汇总（首先读这个） |
| `KFOLD_README.md` | 快速入门指南 |
| `KFOLD_USAGE_GUIDE.md` | 详细参考手册 |
| `KFOLD_TRAINING_GUIDE.md` | 本文件 - 训练指南 |
| `KFOLD_FILE_INDEX.md` | 文件导航 |

---

## 获取帮助

1. **问题**: 不知道从哪开始
   → 答案: 运行 `python kfold_quick_start.py`

2. **问题**: 想看完整文档
   → 答案: 读 `KFOLD_README.md` 或 `KFOLD_USAGE_GUIDE.md`

3. **问题**: 训练遇到错误
   → 答案: 查看"常见问题解决"部分

4. **问题**: 需要自定义集成
   → 答案: 参考 `kfold_data_loader.py` 和Python代码示例

---

**祝训练顺利！** 🎉

如有问题，查看文档或检查脚本中的详细注释。
