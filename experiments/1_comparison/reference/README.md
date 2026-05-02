# 知识蒸馏方法对比项目

该项目是一个大规模的知识蒸馏（Knowledge Distillation, KD）方法对比研究，集合了多种最新蒸馏技术，目的是在统一框架下实现、训练和评估不同的知识蒸馏算法。项目主要针对船舶检测等任务，通过教师-学生网络结构比较各算法的性能。

---

## 📁 目录结构

每个方法都保存在独立的文件夹内，包含模型定义、损失函数、训练脚本和检查点目录。通用目录结构如下：

```
方法名/
├── ResNet_方法名.py       # 教师模型定义
├── Student_方法名.py      # 学生模型定义
├── 方法名Loss.py          # 蒸馏损失函数
├── 方法名.py              # 蒸馏框架实现
├── ReTrain_方法名.py      # 训练脚本（或 Train_方法名.py）
├── Checkpoints_DeepShip_方法名/   # DeepShip 数据集权重
├── Checkpoints_EarShip_方法名/    # EarShip 数据集权重
```

部分方法还包含辅助模块。例如 `diffkd_modules.py` 和 `scheduling_ddim.py` 用于 DiffKD；`SPP.py` 用于 SDD 等。

根目录还有一个 `Test.py`，用于加载训练好的模型并在测试集上评估准确率、F1、ROC-AUC 等指标。

---

## 🚀 支持的蒸馏方法

项目包含 20 多种蒸馏方法，包括但不限于：

- **基础与注意力类**：KD、AT、FT
- **解耦与扩散**：DKD、DiffKD、CAT_KD、LSKD、FreeKD
- **特征级蒸馏**：FSP、NST、PKT、RKD、SP
- **多模态/专用**：MKD、MGD、SDD、OFAKD、ICKD、WSLD、CC、VID、VkD、UATR_KD、NKD、SRRL

每一种方法都有对应的训练和测试代码，以及在 DeepShip 和 EarShip 数据集上的检查点。

---

## 📌 主要组件说明

- **教师模型 (`ResNet_*`)**：常见为 ResNet18，用于生成软标签和特征。
- **学生模型 (`Student_*` 或 `CNNStudent`)**：轻量级网络，目的是通过蒸馏学习教师知识。
- **损失函数 (`*Loss.py`)**：实现各方法的核心蒸馏损失。
- **训练脚本 (`ReTrain_*` / `Train_*`)**：负责数据加载、模型初始化、优化循环及日志记录。
- **测试脚本 (`Test.py`)**：统一的评估程序。
- **检查点 (`Checkpoints_*`)**：存放训练好的模型权重。

---

## 📊 应用与评估

以 `Test.py` 为例，项目使用以下设置：

- 数据预处理：224×224 图像或声谱图
- 评估指标：准确率、F1-score、ROC-AUC
- 支持 GPU 训练，可通过命令行参数配置批量大小、学习率、随机种子等

训练脚本大都支持 `argparse` 参数，示例：

```bash
python ReTrain_KD.py --num_epochs 20 --batch_size 16 --learning_rate 0.001 --gpu 0
```

---

## 🧠 使用建议

1. **选择方法**：进入对应文件夹，查看 `ReTrain_*.py` 或 `Train_*.py`。
2. **设置数据路径**：根据脚本中的默认路径修改为自己的数据集 CSV。
3. **运行训练**：执行训练脚本，模型将保存在对应 `Checkpoints` 目录。
4. **评估性能**：使用 `Test.py` 加载训练好的模型并评估。

---

## 🛠 依赖

- PyTorch
- torchvision
- albumentations
- pandas
- numpy
- scikit-learn
- tensorboardX

请确保在虚拟环境中安装上述依赖。

