# TMSKD 项目完整梳理 - 快速论文撰写指南

## 📋 项目概览

**标题**：TMSKD: Temporal-Memory Structure-Aware Knowledge Distillation for Liquid Neural Network-based Underwater Acoustic Target Recognition

**核心目标**：提出一个基于液态神经网络（LNN）的知识蒸馏框架，用于水下声学目标识别（UATR），在资源受限的水下平台上实现轻量级但高效的目标识别。

---

## 🎯 核心创新点

### 1. **TMSKD 框架架构**（两大核心模块）

#### ① TS-T（Temporal Separation with Learnable Temperature Knowledge Distillation）
- **功能**：时间步级别的知识蒸馏
- **特点**：
  - 采用可学习的温度参数（Learnable Temperature）
  - 对每个时间步的输出进行蒸馏而不是全局pooling
  - 保留了水下声学信号中的瞬间/关键特征
  - 基于KL散度的logit级蒸馏

#### ② MemKD（Memory-Discrepancy Knowledge Distillation）
- **功能**：内存感知的知识蒸馏
- **特点**：
  - 对齐教师与学生的内部动态轨迹
  - 捕捉长期时间进化逻辑
  - 解决异构网络架构间的对齐困难
  - 特别针对LNN的记忆/状态演化设计

### 2. **学生网络架构（LNN AudioCfC）**

```
Input Audio (1D Waveform)
    ↓
[CNN Encoder (5 layers)] - 特征提取
    ↓
[Bidirectional Parallel Cross-Slice CfC (BPCSCfC)]
  - AutoNCP生成的神经回路
  - 双向处理
  - 连续时间ODE动态
    ↓
[DRASP Module] - Dual-Resolution Attentive Statistics Pooling
  - 全局稳定信息
  - 局部瞬间特征
    ↓
[Classification Head]
    ↓
Output (Class Probabilities)
```

**为什么选LNN**：
- 直接处理原始1D波形（无需Mel谱图预处理）
- 生物启发的连续时间动态建模
- 参数效率高，适合资源受限部署
- 对非平稳信号鲁棒性强

### 3. **教师网络**
- 基于Audio_TeacherNet（ViT架构）
- 在Mel谱图上训练
- 在两个数据集上预训练完成
- 训练过程中冻结（不更新参数）

---

## 📊 数据集与实验设置

### 数据集
| 数据集 | 类别数 | 类型 | 特点 |
|--------|--------|------|------|
| **ShipsEar** | 4 | 船舶噪声 | 环境复杂，真实场景 |
| **DeepShip** | 4 | 潜艇类声学 | 特征复杂，长期依赖 |

### 实验方案

#### 方案1：Cross-Validation (50-Fold)
- **文件**：`experiment5_cross_validation.py` 等
- **流程**：
  1. 分层50折交叉验证
  2. 每一折运行完整训练
  3. 收集50个模型的结果
  4. 计算均值和标准差
  
- **关键脚本**：
  - `kfold_cross_validation.py` - 生成K-Fold划分
  - `kfold_data_loader.py` - 加载划分数据
  - `kfold_deepship_integration.py` - DeepShip K-Fold集成
  - `kfold_shipsear_integration.py` - ShipsEar K-Fold集成

#### 方案2：超参数网格搜索（Grid Search）
- **文件**：`train_distillation_deepship.py` 等
- **搜索空间**：
  - Dropout率（p_encoder, p_classifier）
  - 蒸馏权重（alpha, beta, mtskd_weight）
  - 温度参数
  
#### 方案3：Ablation Study（消融实验）
- 验证TS-T的有效性
- 验证MemKD的有效性
- 验证DRASP模块的有效性
- 比较不同蒸馏策略

---

## 🔬 关键实验结果

### 主要成果（根据论文abstract）

| 指标 | ShipsEar | DeepShip |
|------|----------|----------|
| **精度（ACC）** | 92.39% | - |
| **相比基线改进** | +2.36% | - |
| **LNN学生网络改进** | +4.14% | - |

### 对比方法
- **基线（无蒸馏）**
- **KL蒸馏**
- **MSE蒸馏**
- **其他KD方法**

### 性能-复杂度权衡
- 蒸馏模型参数量极少
- 推理速度快
- 适合边缘计算部署

---

## 📁 项目文件结构详解

```
10-LNN/
├── models/                          # 模型定义
│   ├── LNN.py                       # 核心：AudioCfC (LNN学生网络)
│   ├── Audio_TeacherNet.py          # 教师网络
│   ├── distillation.py              # 知识蒸馏框架
│   └── Audio_Teacher_*/             # 预训练教师模型检查点
│
├── configs/                         # 训练配置
│   ├── train_distillation_deepship.yaml    # DeepShip蒸馏配置
│   ├── train_distillation_shipsear.yaml    # ShipsEar蒸馏配置
│   ├── train_LNN_deepship.yaml             # LNN基线配置
│   ├── train_tser_deepship.yaml            # TSER对比配置
│   └── tsne_*.yaml                        # t-SNE可视化配置
│
├── datasets/                        # 数据集处理
│   ├── audio_dataset.py             # 音频数据加载
│   └── audio_dataset_old.py
│
├── experiments/                     # 实验脚本
│   ├── cv/                          # K-Fold交叉验证
│   ├── comparison/                  # 方法对比
│   ├── ablation/                    # 消融实验
│   ├── hyperparameter_analysis/     # 超参数分析
│   └── tsne_analysis/               # 可视化分析
│
├── scripts/                         # 工具脚本
│   └── analyze_log.py               # 日志分析
│
├── train_distillation_*.py          # 主训练脚本（蒸馏）
├── train_LNN_*.py                   # 主训练脚本（基线）
├── experiment5_cross_validation*.py # 交叉验证主程序
├── kfold_*.py                       # K-Fold相关脚本
│
├── results/                         # 实验结果
│   ├── cv_deepship/                 # DeepShip交叉验证结果
│   ├── cv_shipsear/                 # ShipsEar交叉验证结果
│   ├── grid_deepship/               # DeepShip网格搜索结果
│   ├── grid_shipsear/               # ShipsEar网格搜索结果
│   ├── ablation_deepship/           # DeepShip消融结果
│   ├── ablation_shipsear/           # ShipsEar消融结果
│   └── kfold_cv_*/                  # 最终K-Fold结果
│
├── checkpoints/                     # 模型检查点
│   ├── cv_deepship/                 # 交叉验证保存的模型
│   ├── cv_shipsear/
│   ├── grid_deepship/               # 网格搜索保存的模型
│   └── ...
│
├── Manuscript/                      # 论文文件
│   ├── TMSKD.tex                    # 主论文LaTeX
│   ├── Chinese.tex                  # 中文版本
│   ├── references.bib               # 参考文献
│   └── ...
│
└── docs/                            # 文档指南
    ├── cv_kfold.md                  # K-Fold使用指南
    ├── K-FOLD_TRAINING_INTEGRATION_GUIDE.md
    ├── ablation_grid.md
    ├── eval_guide_deepship.md
    ├── TSNE_QUICK_START.md
    └── ...
```

---

## 🚀 快速复现步骤

### 第1步：环境配置
```bash
# 根据requirements安装依赖
pip install torch torchaudio numpy scipy scikit-learn pyyaml ...
```

### 第2步：生成K-Fold划分
```bash
python experiment5_cross_validation.py \
  --data-dir /path/to/ShipsEar_622 \
  --output-dir results/kfold_splits_shipsear
```

### 第3步：运行蒸馏训练（单个fold示例）
```bash
python train_distillation_shipsear.py \
  --config configs/train_distillation_shipsear.yaml \
  --fold 0
```

### 第4步：K-Fold集成训练（运行所有fold）
```bash
python kfold_shipsear_integration.py \
  --config configs/train_distillation_shipsear.yaml \
  --num-folds 50
```

### 第5步：评估与结果汇总
```bash
python scripts/analyze_log.py \
  results/kfold_cv_shipsear/
```

---

## 📈 论文重点内容构成

### 摘要（Abstract）
- ✅ 问题陈述：UATR中的模型压缩问题
- ✅ 核心方案：TMSKD框架 + LNN学生网络
- ✅ 技术创新：TS-T + MemKD + DRASP
- ✅ 实验结果：92.39% ACC on ShipsEar (+2.36%)

### 引言（Introduction）- 3大痛点
1. **异构网络对齐困难**：Teacher用Mel谱图，Student用原始波形
2. **时间信息损失**：Global pooling后丧失关键瞬间特征
3. **轻量化与性能权衡**：如何在参数少的模型上保证精度

### 相关工作（Related Work）- 3个领域
1. **水下声学识别（UATR）**：从特征工程到深度学习
2. **液态神经网络（LNN）**：连续时间ODE动态模型
3. **知识蒸馏（KD）**：从基础KL蒸馏到时间感知蒸馏

### 方法（Methodology）
1. LNN动态基础理论（ODE）
2. TS-T模块详解（时间步级蒸馏）
3. MemKD模块详解（内存对齐）
4. DRASP模块详解（双分辨率池化）
5. 联合损失函数定义

### 实验（Experiments）
1. **数据集**：ShipsEar + DeepShip
2. **对比基线**：无蒸馏、KL蒸馏、MSE蒸馏、其他方法
3. **消融研究**：TS-T + MemKD + DRASP效果
4. **超参敏感性**：温度、蒸馏权重等
5. **复杂度分析**：参数量、计算量、推理速度

### 结果与分析（Results）
- 精度提升对比表
- 精度-效率权衡曲线
- 消融实验结果表
- t-SNE可视化（特征分布）
- 鲁棒性分析（不同噪声水平）

### 结论（Conclusion）
- 总结主要贡献
- 实际应用价值
- 后续研究方向

---

## 🔑 核心参数与超参数

### 蒸馏配置（来自YAML）
```yaml
distillation:
  temperature: 2.0              # KL蒸馏温度
  alpha: 0.3                    # 蒸馏损失权重
  learnable_alpha: true         # 可学习权重
  beta: 0.5                     # 另一权重参数
  learnable_beta: true
  mtskd_weight: 0.5             # MTSKD权重
  memkd_short_weight: 0.5       # 短期MemKD权重
  memkd_long_weight: 1.0        # 长期MemKD权重
```

### 训练配置
```yaml
training:
  num_epochs: 200
  batch_size: 16
  lr: 0.0004                    # 学习率
  weight_decay: 8.0e-05
  seed: 42
```

### 学生网络配置
```yaml
student:
  stu_seq_len: 16               # 序列长度
  p_encoder: 0.35               # 编码器dropout
  p_classifier: 0.1             # 分类器dropout
```

---

## 📊 可视化结果位置

所有可视化结果存储在 `results/` 下：

| 文件夹 | 内容 |
|--------|------|
| `cv_deepship/` | DeepShip交叉验证结果、图表 |
| `cv_shipsear/` | ShipsEar交叉验证结果、图表 |
| `grid_deepship/` | 超参数网格搜索热力图 |
| `grid_shipsear/` | 超参数网格搜索热力图 |
| `ablation_deepship/` | 消融实验柱状图 |
| `ablation_shipsear/` | 消融实验柱状图 |
| `experiment6_*_tsne_perfect/` | t-SNE特征可视化 |
| `hyperparameter_analysis/` | 超参数敏感性分析 |

---

## 💡 论文撰写要点

### ✅ 必须包含的数据/图表
1. **精度对比表**：TMSKD vs 基线 vs 其他KD方法
2. **消融表**：TS-T、MemKD、DRASP各自贡献
3. **t-SNE图**：展示学生网络特征学习质量
4. **训练曲线**：Loss、Accuracy随Epoch变化
5. **参数-性能曲线**：展示轻量化优势
6. **超参敏感性分析**：关键超参的影响

### ✅ 论述框架
1. **动机明确**：为什么要用LNN+KD？
2. **创新清晰**：TS-T和MemKD如何解决问题？
3. **实验充分**：K-Fold CV验证统计显著性
4. **结果翔实**：定量结果 + 定性分析 + 可视化

### ⚠️ 常见问题
- Q: 92.39%是什么基础？A: 50-Fold交叉验证的平均精度
- Q: 教师网络性能如何？A: 查看 `eval_teacher_acc_auc_f1.md`
- Q: 计算复杂度如何计算？A: 参考 `scripts/analyze_log.py`

---

## 🎓 关键文献与理论支撑

### 需要引用的核心概念
1. **知识蒸馏**：Hinton et al., 2015
2. **液态神经网络**：Hasani et al., 2021
3. **时间感知蒸馏**：相关时序模型KD论文
4. **水下声学识别**：领域相关benchmark论文

---

## 📞 快速查询

### 我想找...

**单个Fold的训练脚本** → `train_distillation_shipsear.py`

**K-Fold自动化脚本** → `kfold_shipsear_integration.py`

**对比方法配置** → `configs/train_comparison_distillation_*.yaml`

**消融实验设置** → `experiments/ablation/`

**超参数网格搜索** → `experiments/hyperparameter_analysis/`

**t-SNE可视化代码** → `run_tsne_visualization.py`

**评估指标详解** → `docs/eval_guide_deepship.md`

**结果数据汇总** → `results/validation_results.csv`

---

## ✨ 论文框架模板

```
【标题】TMSKD: 水下声学目标识别的时间-内存结构感知知识蒸馏

【摘要】
- 问题：轻量化+高精度 + 异构网络对齐难
- 方案：TMSKD = TS-T + MemKD + LNN学生
- 结果：92.39% on ShipsEar (+2.36%)

【引言】
- 应用背景：水下监测、海洋安全
- 技术挑战：非平稳信号、部署约束、特征鲁棒性
- 已有方法局限：...

【相关工作】
1. UATR方法演进
2. LNN理论基础
3. KD方法分类

【方法】
1. 问题定义与框架总览
2. LNN基础（ODE）
3. TS-T设计（为什么？怎么做？）
4. MemKD设计（为什么？怎么做？）
5. DRASP设计
6. 联合优化

【实验】
1. 数据集与设置
2. 对比基线
3. 消融研究
4. 超参分析
5. 复杂度对比

【结果与分析】
1. 主要结果表
2. 消融结果表
3. 可视化分析
4. 统计显著性

【结论】
1. 主要贡献总结
2. 实际应用价值
3. 后续研究方向
```

---

**最后更新**：2025年
**项目状态**：论文ready阶段
**建议方向**：整理最终实验数据、制作发表级图表、撰写详细方法说明
