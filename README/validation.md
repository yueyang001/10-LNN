# Validation.py 使用说明

## 功能说明

`validation.py` 是一个自动化验证脚本，可以同时评估教师模型在ShipsEar和DeepShip数据集上的性能，并生成详细的CSV报告。

## 主要特性

✅ **自动评估多个数据集**：同时验证ShipsEar和DeepShip的validation和test集  
✅ **正确的标签映射**：修复了audio_dataset.py中的标签映射问题  
✅ **完整的评估指标**：Accuracy, F1-Macro/Micro/Weighted, AUC-OVO/OVR  
✅ **控制台输出**：实时打印验证结果  
✅ **CSV导出**：自动生成CSV文件保存所有结果  
✅ **汇总表格**：最后打印所有评估结果的汇总表

## 使用方法

### 基本用法

```bash
conda run -n UATR python validation.py
```

### 指定参数

```bash
conda run -n UATR python validation.py \
  --batch_size 32 \
  --gpu 4 \
  --output_csv validation_results.csv
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_type` | `wav_t` | 数据类型（用于教师模型） |
| `--batch_size` | `32` | 批次大小 |
| `--num_workers` | `4` | 数据加载线程数 |
| `--gpu` | `4` | 使用的GPU ID |
| `--output_csv` | `validation_results.csv` | 结果保存的CSV文件路径 |

## 评估配置

脚本会自动评估以下4个配置：

### ShipsEar模型
1. **Validation Set**：ShipsEar_622/validation
2. **Test Set**：ShipsEar_622/test

### DeepShip模型
3. **Validation Set**：DeepShip_622/validation
4. **Test Set**：DeepShip_622/test

## 输出示例

### 控制台输出

```
================================================================================
EVALUATION 1/4
================================================================================
Model: Student.pth
Dataset: /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622
Data flag: validation
================================================================================

Using device: cuda:4
Model loaded from: /media/hdd1/fubohan/Code/UATR/models/Audio_Teacher_ShipsEar_622/checkpoints/Student.pth
Dataset: validation
Dataset size: 749
Classes: ['A', 'B', 'C', 'D', 'E']

Processing batch [10/24]
Processing batch [20/24]

================================================================================
VALIDATION RESULTS
================================================================================
Accuracy: 94.6595%

F1 Score:
  Macro:   94.7255%
  Micro:   94.6595%
  Weighted: 94.6620%

AUC Score:
  OVO (One-vs-One):  99.6283%
  OVR (One-vs-Rest): 99.5505%

Per-class Accuracy:
  Class 0: 98.39%
  Class 1: 97.55%
  Class 2: 91.52%
  Class 3: 91.26%
  Class 4: 98.68%
================================================================================
```

### 汇总表格

```
================================================================================
SUMMARY TABLE
================================================================================
model_path                                            dataset          data_flag   accuracy  f1_macro  f1_weighted  auc_ovo  auc_ovr
Student.pth                                         ShipsEar_622     validation    94.6595   94.7255    94.6620   99.6283  99.5505
Student.pth                                         ShipsEar_622     test          95.2345   95.1200    95.2100   99.8123  99.7890
Student.pth                                         DeepShip_622     validation    92.1234   91.8900    92.0500   98.7654  98.7432
Student.pth                                         DeepShip_622     test          93.4567   93.2100    93.3400   99.0123  99.0012
================================================================================
```

### CSV文件格式

```csv
model_path,dataset,data_flag,num_samples,accuracy,f1_macro,f1_micro,f1_weighted,auc_ovo,auc_ovr,per_class_accuracy
Student.pth,ShipsEar_622,validation,749,94.6595,94.7255,94.6595,94.6620,99.6283,99.5505,[98.39, 97.55, 91.52, 91.26, 98.68]
Student.pth,ShipsEar_622,test,749,95.2345,95.1200,95.2345,95.2100,99.8123,99.7890,[98.50, 97.80, 92.10, 91.80, 98.90]
Student.pth,DeepShip_622,validation,10632,92.1234,91.8900,92.1234,92.0500,98.7654,98.7432,[92.50, 91.80, 91.20, 92.00]
Student.pth,DeepShip_622,test,10632,93.4567,93.2100,93.4567,93.3400,99.0123,99.0012,[93.80, 92.90, 93.10, 93.00]
```

## 评估指标说明

### Accuracy（准确率）
整体分类正确的样本比例

### F1 Score
- **Macro**: 各类别F1的算术平均（不考虑类别不平衡）
- **Micro**: 基于所有样本的总体指标计算F1
- **Weighted**: 按类别样本数加权计算F1

### AUC Score
- **OVO (One-vs-One)**: 一对一策略的AUC
- **OVR (One-vs-Rest)**: 一对多策略的AUC

### Per-class Accuracy
每个类别的单独准确率

## 修改评估配置

如需修改评估的模型或数据集，编辑 `main()` 函数中的 `evaluations` 列表：

```python
evaluations = [
    {
        'model_path': '/path/to/model.pth',
        'num_classes': 5,
        'data_dir': '/path/to/dataset',
        'data_flag': 'validation'
    },
    # 添加更多配置...
]
```

## 注意事项

1. **GPU内存**：如果遇到OOM错误，减小 `--batch_size`
2. **数据集路径**：确保数据集路径正确存在
3. **标签映射**：脚本已修复audio_dataset.py中的标签映射问题
4. **运行时间**：评估4个配置大约需要10-20分钟（取决于GPU性能）

## 输出文件

- **控制台**：实时打印每个评估的详细结果
- **CSV文件**：保存所有评估结果，方便后续分析
- **汇总表**：在控制台最后打印，方便快速对比

## 示例命令

```bash
# 使用默认参数
conda run -n UATR python validation.py

# 指定GPU和batch size
conda run -n UATR python validation.py --gpu 0 --batch_size 16

# 指定输出文件名
conda run -n UATR python validation.py --output_csv results_$(date +%Y%m%d).csv

# 组合使用
conda run -n UATR python validation.py --batch_size 64 --gpu 4 --output_csv validation_final.csv
```
