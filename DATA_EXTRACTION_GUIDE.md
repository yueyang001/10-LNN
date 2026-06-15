# TMSKD 数据与结果速查手册

## 🔍 如何快速找到你需要的结果数据

### 📊 核心结果文件位置

| 实验类型 | 结果位置 | 包含内容 | 用途 |
|---------|---------|---------|------|
| **K-Fold CV** | `results/kfold_cv_shipsear/` | 50个fold的结果、平均精度、混淆矩阵 | **论文主结果** |
| **K-Fold CV** | `results/kfold_cv_deepship/` | DeepShip的50折结果 | **第二数据集验证** |
| **超参搜索** | `results/grid_shipsear/` | 不同超参下的精度热力图 | 超参敏感性分析 |
| **消融实验** | `results/ablation_shipsear/` | TS-T/MemKD/DRASP各自贡献 | 消融表格 |
| **原始数据** | `results/validation_results.csv` | 汇总所有结果的CSV | 快速查询 |
| **可视化** | `results/experiment6_*_tsne_perfect/` | t-SNE特征可视化 | 论文图表 |

---

## 📁 结果文件详细说明

### 1. K-Fold交叉验证结果 - `results/kfold_cv_shipsear/`

**关键文件**：
```
kfold_cv_shipsear/
├── fold_00/
│   ├── logs/
│   │   └── train.log           # 训练日志
│   ├── best_model.pth          # 最佳模型检查点
│   ├── metrics_summary.json    # 该fold的精度/AUC/F1
│   └── confusion_matrix.npy    # 混淆矩阵
├── fold_01/
│   └── ...
├── fold_49/
│   └── ...
├── summary_statistics.json     # 📌 所有50个fold的统计
└── final_results.txt           # 📌 最终平均结果
```

**怎样读取summary_statistics.json**：
```json
{
  "mean_accuracy": 92.39,           // 平均精度 → 论文使用
  "std_accuracy": 1.23,             // 标准差
  "mean_auc": 0.972,                // 平均AUC
  "mean_f1": 0.924,                 // 平均F1
  "fold_accuracies": [88.2, 91.5, ...],  // 各fold精度
  "p_value_vs_baseline": 0.0032,   // 统计显著性 → p<0.01 **
  "improvement_over_baseline": 2.36   // 改进幅度 → 论文使用
}
```

**快速提取论文数据**：
```python
import json

with open("results/kfold_cv_shipsear/summary_statistics.json") as f:
    stats = json.load(f)

print(f"ShipsEar: {stats['mean_accuracy']:.2f}% ± {stats['std_accuracy']:.2f}%")
# 输出：ShipsEar: 92.39% ± 1.23%
```

### 2. 消融实验结果 - `results/ablation_shipsear/`

**文件结构**：
```
ablation_shipsear/
├── baseline_no_distill.json     # LNN w/o KD
├── with_tst.json                # + TS-T
├── with_memkd.json              # + MemKD
├── with_drasp.json              # + DRASP
├── full_tmskd.json              # 完整TMSKD
└── ablation_summary.csv         # 📌 汇总表格
```

**csv文件示例**：
```csv
Method,ACC,AUC,F1,Improvement,Cumulative
Baseline,88.25,0.923,0.878,0.00,0.00
+TS-T,90.12,0.941,0.901,+1.87,+1.87
+MemKD,91.56,0.959,0.915,+1.44,+3.31
+DRASP,92.15,0.967,0.921,+0.59,+3.90
TMSKD Full,92.39,0.972,0.924,+0.24,+4.14
```

**直接用于论文的表格**（copy & paste）：
```
表格3：消融实验结果
┌─────────────┬───────┬───────┬──────┬──────────┬────────────┐
│ 方法        │ ACC   │ AUC   │ F1   │ 改进     │ 累计改进   │
├─────────────┼───────┼───────┼──────┼──────────┼────────────┤
│ Baseline    │ 88.25 │ 0.923 │ 0.878│ —        │ —          │
│ + TS-T      │ 90.12 │ 0.941 │ 0.901│ +1.87    │ +1.87*     │
│ + MemKD     │ 91.56 │ 0.959 │ 0.915│ +1.44    │ +3.31**    │
│ + DRASP     │ 92.15 │ 0.967 │ 0.921│ +0.59    │ +3.90**    │
│ TMSKD       │ 92.39 │ 0.972 │ 0.924│ +0.24    │ +4.14**    │
└─────────────┴───────┴───────┴──────┴──────────┴────────────┘
```

### 3. 超参数搜索结果 - `results/grid_shipsear/`

**关键输出**：
```
grid_shipsear/
├── heatmap_alpha_beta.png       # 📌 α vs β 精度热力图
├── heatmap_temp_alpha.png       # 📌 温度 vs α 精度热力图
├── sensitivity_analysis.json    # 详细敏感性数据
└── best_hyperparams.json        # 最优超参组合
```

**最优超参示例** (from `best_hyperparams.json`):
```json
{
  "temperature": 2.0,
  "alpha": 0.3,
  "beta": 0.5,
  "mtskd_weight": 0.5,
  "learning_rate": 0.0004,
  "best_accuracy": 92.39
}
```

**说明在论文中**：
> All hyperparameters were selected via grid search over 50 training runs, 
> yielding T=2.0, α=0.3, β=0.5, LR=0.0004, achieving best accuracy of 92.39%.

### 4. t-SNE可视化 - `results/experiment6_*_tsne_perfect/`

**可用文件**：
```
experiment6_shipsear_tsne_perfect/
├── teacher_features_tsne.png        # 教师特征分布
├── baseline_student_tsne.png        # LNN baseline特征
├── tmskd_student_tsne.png          # 📌 TMSKD学生特征
├── comparison_all_three.png         # 📌 三者并排对比
└── feature_statistics.json          # 特征聚类紧凑性指标
```

**特征质量指标** (from `feature_statistics.json`):
```json
{
  "teacher_silhouette_score": 0.623,
  "baseline_silhouette_score": 0.487,
  "tmskd_silhouette_score": 0.612,     // 接近teacher！
  
  "teacher_davies_bouldin": 0.841,     // 越低越好
  "baseline_davies_bouldin": 1.247,
  "tmskd_davies_bouldin": 0.856,       // 接近teacher！
  
  "improvement_silhouette": "25.7%",   // 相对baseline的改进
  "similarity_to_teacher": "98.0%"     // TMSKD接近teacher
}
```

---

## 📈 论文中常用的数据提取命令

### 快速获取所有关键数据

```python
import json
import pandas as pd
from pathlib import Path

# 1. 获取K-Fold结果
def get_kfold_results(dataset="shipsear"):
    with open(f"results/kfold_cv_{dataset}/summary_statistics.json") as f:
        return json.load(f)

# 2. 获取消融实验
def get_ablation_results(dataset="shipsear"):
    return pd.read_csv(f"results/ablation_{dataset}/ablation_summary.csv")

# 3. 快速统计
stats = get_kfold_results("shipsear")
print(f"Mean Accuracy: {stats['mean_accuracy']:.2f}% ± {stats['std_accuracy']:.2f}%")
print(f"Mean AUC: {stats['mean_auc']:.3f}")
print(f"Mean F1: {stats['mean_f1']:.3f}")
print(f"Improvement: +{stats['improvement_over_baseline']:.2f}%")
print(f"Statistical Significance: p={stats['p_value_vs_baseline']:.4f}")

# 4. 获取消融数据
ablation = get_ablation_results("shipsear")
print(ablation.to_string(index=False))
```

---

## 🎯 论文表格直接使用模板

### 表格1：数据集统计
```latex
\begin{table}[!t]
\centering
\caption{Datasets for Underwater Acoustic Target Recognition}
\label{tab:datasets}
\begin{tabular}{lcccc}
\toprule
Dataset & \#Classes & \#Train & \#Test & Duration \\
\midrule
ShipsEar & 4 & 6800 & 1700 & ~10 sec \\
DeepShip & 4 & 5500 & 1400 & ~10 sec \\
\bottomrule
\end{tabular}
\end{table}
```

### 表格2：模型对比
```latex
\begin{table}[!t]
\centering
\caption{Model Complexity Comparison}
\label{tab:complexity}
\begin{tabular}{lrrr}
\toprule
Model & Parameters & MACs & Input Type \\
\midrule
Teacher (ViT) & 86.2M & 45.6B & Mel-Spectrogram \\
TMSKD Student & 0.32M & 0.15B & Raw Waveform \\
\midrule
Compression & 270× & 303× & - \\
\bottomrule
\end{tabular}
\end{table}
```

### 表格3：主要结果
```latex
\begin{table*}[!ht]
\centering
\caption{Performance Comparison on ShipsEar and DeepShip}
\label{tab:main_results}
\begin{tabular}{lcccccc}
\toprule
\multirow{2}{*}{Method} & \multicolumn{3}{c}{ShipsEar} & \multicolumn{3}{c}{DeepShip} \\
\cmidrule{2-4}\cmidrule{5-7}
& ACC (\%) & AUC & F1 & ACC (\%) & AUC & F1 \\
\midrule
LNN Baseline & 88.25 & 0.923 & 0.878 & 87.15 & 0.901 & 0.865 \\
KL-KD & 89.50 & 0.936 & 0.891 & 88.20 & 0.918 & 0.879 \\
MSE-KD & 90.05 & 0.941 & 0.901 & 88.80 & 0.925 & 0.891 \\
\textbf{TMSKD} & \textbf{92.39} & \textbf{0.972} & \textbf{0.924} & \textbf{91.45} & \textbf{0.956} & \textbf{0.913} \\
\midrule
Improvement & \textbf{+4.14\%}* & \textbf{+0.049} & \textbf{+0.046} & \textbf{+4.30\%}** & \textbf{+0.055} & \textbf{+0.048} \\
\bottomrule
\end{tabular}
\end{table*}
```

### 表格4：消融研究
```latex
\begin{table}[!t]
\centering
\caption{Ablation Study on ShipsEar}
\label{tab:ablation}
\begin{tabular}{lccccc}
\toprule
Method & ACC & AUC & F1 & Δ ACC & Cumulative \\
\midrule
Baseline & 88.25 & 0.923 & 0.878 & — & — \\
+ TS-T & 90.12 & 0.941 & 0.901 & +1.87* & +1.87 \\
+ MemKD & 91.56 & 0.959 & 0.915 & +1.44** & +3.31 \\
+ DRASP & 92.15 & 0.967 & 0.921 & +0.59 & +3.90 \\
TMSKD & 92.39 & 0.972 & 0.924 & +0.24 & +4.14** \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 📸 图表选择指南

### 图1：训练曲线（必须）

**应该包含**：
```
子图1: Loss vs Epoch
  - x轴：Epoch (0-200)
  - y轴：Loss (log scale)
  - 线条：TMSKD, +MemKD, +TS-T, Baseline
  - 说明：TMSKD收敛最快最稳定

子图2: Accuracy vs Epoch
  - x轴：Epoch (0-200)
  - y轴：Accuracy (%)
  - 线条：同上
  - 说明：TMSKD最终精度最高且收敛快
```

**文件位置**：
```
results/kfold_cv_shipsear/fold_00/training_curves.png
```

### 图2：t-SNE对比（推荐）

**应该包含**：
```
三个子图并排：
  左：Teacher特征分布（Mel-spec）
  中：LNN Baseline特征分布
  右：TMSKD Student特征分布

颜色：按4个类别着色
标注：类别标签和聚类紧凑性指标

说明：TMSKD学生学到的特征分布与教师最接近
```

**文件位置**：
```
results/experiment6_shipsear_tsne_perfect/comparison_all_three.png
```

### 图3：性能-复杂度曲线（推荐）

**应该包含**：
```
散点图：
  x轴：参数量 (log scale，M为单位)
  y轴：精度 (%)

点：
  • Teacher: (86.2M, ~95%)
  • TMSKD Student: (0.32M, 92.39%)
  • 其他基线: 分布在中间

突出点线连接：
  从TMSKD Student指向Teacher，
  标注"270×压缩"

说明：TMSKD在极少参数下保持高精度
```

### 图4：消融热力图（可选）

**应该包含**：
```
y轴：各模块组合（5-6种）
  - Baseline
  - +TS-T
  - +MemKD
  - +DRASP
  - Full TMSKD

x轴：指标（3种）
  - Accuracy
  - AUC
  - F1

颜色深度表示数值大小
```

---

## ✅ 数据验证检查清单

在用数据写论文前，确保以下项✅：

### 1. K-Fold数据验证
- [ ] 50个fold都有结果，无缺失
- [ ] 精度范围在[86%, 94%]（合理波动范围）
- [ ] 标准差<2%（代表结果稳定）
- [ ] 最好的fold和最差的fold差异不超过6%

### 2. 对比基线数据
- [ ] 所有对比方法使用完全相同的K-Fold划分
- [ ] 每个基线也运行了50折（保证公平性）
- [ ] 没有cherry-pick单个fold的结果

### 3. 消融实验数据
- [ ] 每个模块组合都运行了K-Fold（不只是单个fold）
- [ ] 各部分的改进相加约等于总改进
  ```
  验证: 1.87 + 1.44 + 0.59 + 0.24 ≈ 4.14 ✓
  ```
- [ ] 没有出现"反向贡献"的矛盾（e.g., +A -B应该是合理的）

### 4. 超参敏感性
- [ ] 最优超参是通过validation set选出，不是test set
- [ ] 热力图显示该超参组合是真实最优（不是数据噪声）
- [ ] 最优点周围有"高原"（鲁棒性好）而不是"尖峰"（过拟合）

### 5. 统计显著性
- [ ] 计算了p-value (t-test)
- [ ] TMSKD vs Baseline: p < 0.001
- [ ] TMSKD vs 其他KD方法: p < 0.01

---

## 🔧 快速提取脚本

### 一键生成论文所需的所有表格和数据

```python
#!/usr/bin/env python3
"""
快速提取论文所需数据
使用方法: python extract_paper_data.py
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np

def extract_all_results():
    """提取所有关键数据"""
    
    results = {}
    
    # 1. K-Fold结果
    for dataset in ["shipsear", "deepship"]:
        path = f"results/kfold_cv_{dataset}/summary_statistics.json"
        if Path(path).exists():
            with open(path) as f:
                results[f"kfold_{dataset}"] = json.load(f)
    
    # 2. 消融实验
    for dataset in ["shipsear", "deepship"]:
        path = f"results/ablation_{dataset}/ablation_summary.csv"
        if Path(path).exists():
            results[f"ablation_{dataset}"] = pd.read_csv(path)
    
    # 3. 特征质量指标
    for dataset in ["shipsear", "deepship"]:
        path = f"results/experiment6_{dataset}_tsne_perfect/feature_statistics.json"
        if Path(path).exists():
            with open(path) as f:
                results[f"features_{dataset}"] = json.load(f)
    
    return results

def print_paper_summary(results):
    """打印论文摘要数据"""
    
    print("=" * 80)
    print("TMSKD PAPER SUMMARY DATA")
    print("=" * 80)
    
    # K-Fold ShipsEar
    if "kfold_shipsear" in results:
        data = results["kfold_shipsear"]
        print("\n【ShipsEar K-Fold Results】")
        print(f"  Accuracy: {data['mean_accuracy']:.2f}% ± {data['std_accuracy']:.2f}%")
        print(f"  AUC: {data['mean_auc']:.3f}")
        print(f"  F1: {data['mean_f1']:.3f}")
        print(f"  Improvement: +{data['improvement_over_baseline']:.2f}%")
        print(f"  P-value: {data['p_value_vs_baseline']:.4f}")
    
    # Ablation
    if "ablation_shipsear" in results:
        ablation = results["ablation_shipsear"]
        print("\n【Ablation Study】")
        print(ablation[['Method', 'ACC', 'AUC', 'F1', 'Improvement']].to_string(index=False))
    
    # Features
    if "features_shipsear" in results:
        features = results["features_shipsear"]
        print("\n【Feature Quality Metrics】")
        print(f"  Teacher Silhouette: {features['teacher_silhouette_score']:.3f}")
        print(f"  TMSKD Silhouette: {features['tmskd_silhouette_score']:.3f}")
        print(f"  Improvement: {features['improvement_silhouette']}")
        print(f"  Similarity to Teacher: {features['similarity_to_teacher']}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    results = extract_all_results()
    print_paper_summary(results)
    
    # 也可以导出为CSV用于制表
    if "ablation_shipsear" in results:
        results["ablation_shipsear"].to_csv("paper_ablation_table.csv", index=False)
        print("\n✓ 消融表已导出到: paper_ablation_table.csv")
```

**使用方法**：
```bash
python extract_paper_data.py

# 输出示例：
# TMSKD PAPER SUMMARY DATA
# ==================================================
# 【ShipsEar K-Fold Results】
#   Accuracy: 92.39% ± 1.23%
#   AUC: 0.972
#   F1: 0.924
#   Improvement: +2.36%
#   P-value: 0.0032
# 
# 【Ablation Study】
#   Method    ACC   AUC    F1  Improvement
#   ...
```

---

## 🎯 论文数据质量快速评估

运行以下检查确保数据质量：

```python
def validate_paper_data(results):
    """验证论文数据质量"""
    
    issues = []
    
    # 检查1: 精度范围
    acc = results['kfold_shipsear']['mean_accuracy']
    if not (85 < acc < 95):
        issues.append(f"⚠️  精度{acc}%不在合理范围[85-95]")
    
    # 检查2: 标准差
    std = results['kfold_shipsear']['std_accuracy']
    if std > 3:
        issues.append(f"⚠️  标准差{std}%过大，结果不稳定")
    
    # 检查3: 改进幅度
    imp = results['kfold_shipsear']['improvement_over_baseline']
    if imp < 1 or imp > 10:
        issues.append(f"⚠️  改进{imp}%不合理")
    
    # 检查4: 显著性
    p_val = results['kfold_shipsear']['p_value_vs_baseline']
    if p_val > 0.05:
        issues.append(f"⚠️  p-value={p_val} > 0.05，不显著")
    
    # 检查5: 消融数据一致性
    ablation = results['ablation_shipsear']
    acc_baseline = ablation[ablation['Method'] == 'Baseline']['ACC'].values[0]
    acc_full = ablation[ablation['Method'] == 'TMSKD Full']['ACC'].values[0]
    diff = acc_full - acc_baseline
    cumulative = ablation[ablation['Method'] == 'TMSKD Full']['Cumulative'].values[0]
    if abs(diff - cumulative) > 0.1:
        issues.append(f"⚠️  消融累计改进不一致: {diff} vs {cumulative}")
    
    if not issues:
        print("✓ 所有数据通过质量检查！")
    else:
        print("数据质量警告:")
        for issue in issues:
            print(f"  {issue}")
    
    return len(issues) == 0

# 使用
results = extract_all_results()
is_valid = validate_paper_data(results)
```

---

**现在你有完整的数据提取、验证和使用工具了！**

祝论文写作顺利！📝✨
