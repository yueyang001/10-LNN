# BiTAL 学生网络更名修改说明

## 1. 统一命名

- 英文名称：**Bidirectional Temporal-Aware Liquid Network**
- 英文缩写：**BiTAL**
- 中文名称：**双向时序感知液态神经网络**
- 使用原则：本文提出的学生网络统一称为 **BiTAL**；“Liquid Neural Networks / 液态神经网络”仅在介绍通用技术类别时保留，不再使用旧缩写 `LNN` 或 `LNNs`。

## 2. 英文稿修改位置

文件：`TMSKDv1.tex`

| 当前行号 | 位置 | 修改内容 |
|---:|---|---|
| 28、33 | 论文标题、页眉短标题 | 将 “Liquid Neural Network-based” 改为 “BiTAL-based”。 |
| 39 | Abstract | 首次完整定义 “Bidirectional Temporal-Aware Liquid Network (BiTAL)”，并将学生模型名称统一为 BiTAL。 |
| 44 | Index Terms | 将通用的 “liquid neural networks” 调整为模型专名关键词 “bidirectional temporal-aware liquid network”。 |
| 53 | 图 1 说明文字 | 将 “liquid neural network-based student” 改为 “BiTAL student”。 |
| 59、61 | Introduction | 在正文首次出现处完整定义 BiTAL；后续将旧学生网络缩写替换为 BiTAL。 |
| 68 | Contributions | 将第一项贡献明确表述为设计 BiTAL 学生网络，并保持 BPCSCfC、DRASP 等结构说明不变。 |
| 80–83 | Related Work | 保留 Liquid Neural Networks 作为技术类别名称，但移除旧缩写 `LNN/LNNs`，避免与 BiTAL 专名混用。 |
| 91、96 | Methodology 开头及总体框架图注 | 将学生网络表述统一为 “BiTAL-based structure” 和 “BiTAL student network”。 |
| 129、136、141 | 学生网络小节、结构图图注、算法标题 | 明确标注为 BiTAL 学生网络。 |
| 217 | MemKD 小节 | 将 MemKD 的受指导学生网络改称 BiTAL。 |
| 325 | Overall Training Loss | 将学生网络改称 BiTAL，并合并原稿中重复的 “LNN LNN structure” 表述。 |
| 394 | KD 方法对比表 | 将基线表项 “Student (LNN)” 改为 “Student (BiTAL)”。 |
| 438、464、470 | SOTA 对比正文与表格 | 将 `LNN-3`、`LNN (w/ KD)` 等实验标签统一为 BiTAL。 |
| 777、868 | BPCSCfC 与 CfC 容量分析 | 将学生网络指代统一为 BiTAL。 |
| 973 | Conclusion | 将学生模型及其骨干表述统一为 BiTAL，并保留 BPCSCfC 和 DRASP 的功能说明。 |

## 3. 中文稿修改位置

文件：`Chinese.tex`

| 当前行号 | 位置 | 修改内容 |
|---:|---|---|
| 26 | 论文标题 | 与英文稿同步改为 “BiTAL-based”。 |
| 37 | 摘要 | 首次给出“双向时序感知液态神经网络（Bidirectional Temporal-Aware Liquid Network, BiTAL）”完整定义，并统一学生模型名称。 |
| 40、42 | 注释保留的摘要草稿 | 同步替换旧模型名称，防止后续恢复注释内容时重新引入旧称。 |
| 44 | 关键词 | 将“液态神经网络”改为“双向时序感知液态神经网络”。 |
| 50 | 图 1 说明文字 | 将轻量级学生网络改称 BiTAL。 |
| 58、60 | 引言 | 区分通用液态神经网络范式与本文提出的 BiTAL，并在中文正文首次出现处给出中英文全称。 |
| 71 | 主要贡献 | 将学生网络架构明确命名为 BiTAL。 |
| 88–94 | 相关工作 | 保留“液态神经网络”作为通用技术类别，并移除旧缩写 `LNN`。 |
| 107、112 | 方法概述及总体框架图注 | 将水声识别结构和学生网络统一称为 BiTAL。 |
| 150–152 | 学生网络小节 | 将英文注释、小节标题和正文统一为 BiTAL 架构。 |
| 159、164 | 学生网络结构图图注、算法标题 | 明确标注 BiTAL 学生网络。 |
| 429 | 总损失函数 | 将闭式连续时间学生结构改称 BiTAL。 |
| 519 | KD 方法对比表 | 将 “Student (LNN)” 改为 “Student (BiTAL)”。 |
| 598、599 | 注释保留的复杂度表项 | 将旧表项名称同步改为 BiTAL。 |
| 606、608、637 | SOTA 对比正文与表格 | 将 `LNN-3`、`LNN (w/ KD)` 等实验标签统一为 BiTAL。 |
| 662、904 | BPCSCfC 与 CfC 容量分析 | 将学生网络指代统一为 BiTAL。 |
| 1031、1032 | 结论 | 将学生模型及其主干表述统一为 BiTAL，并保留 BPCSCfC、DRASP 的结构说明。 |

## 4. 一致性检查结果

- 两份 `.tex` 中已无作为模型名称出现的 `LNN`、`LNNs` 或 `LNN-3`。
- 英文稿与中文稿均在摘要和引言首次出现处给出 BiTAL 的完整定义。
- 表格、图注、算法标题、实验分析和结论中的模型名称已同步。
- 图片文件名、LaTeX 内部标签（如 `eq:lnn_ode`）以及代码文件名未改动，因为它们属于内部资源引用，直接更名可能导致编译或代码引用失效。

## 5. 尚待处理的图内文字

- `figures/model_structure_v2.0.png` 的学生网络框内仍显示 `Student: lightweight LNN`。
- 已定位其可编辑源稿为 `structure_image/overall.pptx` 的第 1 页，目标文字同为 `Student: lightweight LNN`，建议改为 `Student: BiTAL` 后重新导出并覆盖上述 PNG。
- 本次未直接修改该 PPTX 或 PNG：当前环境缺少演示文稿编辑所需的 `@oai/artifact-tool` 运行时，无法完成源稿编辑后的渲染与版式验证。为避免破坏复杂公式、连线和图形布局，没有采用直接修改压缩包内部 OOXML 或生成式重绘的替代方案。
