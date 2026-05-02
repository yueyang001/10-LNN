# DeepShip 学生模型评估脚本

本目录包含用于评估 DeepShip 数据集上训练的学生模型性能的脚本。

## 文件说明

- `evaluate_student_deepship.py` - 单个模型评估脚本
- `batch_evaluate_ablation_deepship.py` - 批量评估脚本(评估所有 ablation_deepship 中的模型)

## 环境要求

所有依赖已安装在 `UATR` conda 环境中。主要依赖:
- PyTorch 2.6.0
- torchaudio 2.6.0
- torchvision 0.21.0
- transformers 4.57.3
- scikit-learn 1.7.2
- pandas 2.3.3

## 使用方法

### 1. 评估单个模型

```bash
# 激活 UATR 环境
conda activate UATR

# 运行评估脚本
python evaluate_student_deepship.py \
    --checkpoint "/path/to/best_student.pth" \
    --config "/path/to/config.yaml" \
    --batch_size 32 \
    --gpu 0 \
    --dataset test  # 可选: train, validation, test
```

**参数说明:**
- `--checkpoint`: best_student.pth 文件的路径
- `--config`: 训练配置文件路径
- `--batch_size`: 批次大小(默认 32)
- `--gpu`: 使用的 GPU ID(默认 0)
- `--dataset`: 评估的数据集,可选 'train', 'validation', 'test'(默认 'test')

**输出:**
- 控制台输出: Accuracy, F1-Score (Macro/Micro/Weighted), AUC (OVO/OVR), Confusion Matrix, Classification Report
- 结果文件: 自动保存在 checkpoint 同目录下,命名为 `evaluation_results_{dataset}.txt`

### 2. 批量评估所有模型

```bash
# 激活 UATR 环境
conda activate UATR

# 运行批量评估脚本
python batch_evaluate_ablation_deepship.py
```

**功能:**
- 自动查找 `checkpoints/ablation_deepship/` 下所有包含 `best_student.pth` 的目录
- 逐个评估所有模型
- 生成汇总报告,按 Accuracy 排名
- 保存详细的评估结果到每个模型的目录
- 生成汇总文件 `evaluation_summary_{dataset}.txt`

**可修改的配置:**
在脚本开头可以修改以下配置:
```python
base_dir = '/media/hdd1/fubohan/Code/UATR/checkpoints/ablation_deepship'  # 模型目录
config_file = '/media/hdd1/fubohan/Code/UATR/configs/train_distillation_deepship.yaml'  # 配置文件
batch_size = 32  # 批次大小
gpu_id = '0'  # GPU ID
dataset = 'test'  # 数据集
conda_env = 'UATR'  # conda 环境
```

## 评估指标说明

### 1. Accuracy (准确率)
- 整体分类准确率

### 2. F1-Score
- **Macro**: 计算每个类别的 F1 分数,然后取平均(不考虑类别不平衡)
- **Micro**: 计算所有类别的总体指标后再计算 F1(考虑类别不平衡)
- **Weighted**: 按照每个类别的支持度加权计算 F1

### 3. AUC (Area Under Curve)
- **OVO (One-vs-One)**: 一对一策略,对每对类别计算 AUC 然后平均
- **OVR (One-vs-Rest)**: 一对多策略,每个类别对其余类别计算 AUC

### 4. Confusion Matrix (混淆矩阵)
- 显示每个类别的预测准确性和混淆情况

### 5. Classification Report (分类报告)
- 每个类别的 Precision, Recall, F1-Score
- 整体的 macro avg 和 weighted avg

## 示例输出

```
============================================================
EVALUATION RESULTS
============================================================

Accuracy: 86.9827%

F1 Score:
  Macro:   86.9924%
  Micro:   86.9827%
  Weighted: 87.0198%

AUC Score:
  OVO (One-vs-One):   97.7024%
  OVR (One-vs-Rest):  97.7055%

Confusion Matrix:
[[2354   96   79   84]
 [ 146 2398  205   10]
 [  83  257 2174   47]
 [ 177   44  156 2322]]

Classification Report:
               precision    recall  f1-score   support

passengership     0.8529    0.9009    0.8762      2613
       tanker     0.8580    0.8692    0.8635      2759
        cargo     0.8317    0.8489    0.8402      2561
          tug     0.9428    0.8603    0.8997      2699

     accuracy                         0.8698     10632
    macro avg     0.8713    0.8698    0.8699     10632
 weighted avg     0.8719    0.8698    0.8702     10632
```

## 注意事项

1. 确保 `best_student.pth` 文件包含正确的模型权重
2. 配置文件中的数据集路径必须正确
3. 确保使用的 GPU 可用且有足够的显存
4. 批量评估可能需要较长时间,建议在后台运行

## 故障排除

### 常见错误

1. **ModuleNotFoundError**: 确保已激活 UATR 环境并安装了所有依赖
2. **CUDA out of memory**: 减小 `--batch_size`
3. **File not found**: 检查 checkpoint 和 config 文件路径是否正确
4. **Data loading error**: 检查配置文件中的数据集路径

### 联系方式

如有问题,请联系项目维护者。
