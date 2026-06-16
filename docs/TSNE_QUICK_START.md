# 四方法 t-SNE 可视化快速使用

本流程用于在同一批验证样本上比较四个学生模型 checkpoint 的 t-SNE 分布。当前默认输出为四张独立小图，便于在 LaTeX 中自行排成 2x2 的单栏或半栏版式。

四个方法通常对应：

1. `Baseline`：未蒸馏学生模型。
2. `Best`：本文提出的最优蒸馏方法。
3. `Logits KD`：基于 logits 的蒸馏方法。
4. `Feature KD`：基于 feature 的蒸馏方法。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `run_tsne_visualization.py` | 快速启动脚本，会调用主绘图脚本 |
| `experiments/tsne_analysis/batch_tsne_visualization.py` | 主脚本，读取四个 checkpoint 并生成 t-SNE 图 |
| `configs/tsne_four_methods_deepship.yaml` | DeepShip 四方法配置 |
| `configs/tsne_four_methods_shipsear.yaml` | ShipsEar 四方法配置 |

## 第一步：修改 YAML

打开对应配置文件，例如：

```bash
configs/tsne_four_methods_deepship.yaml
```

重点检查以下部分：

```yaml
output_dir: results/tsne_four_methods_deepship_v2
feature_layer: logits
max_samples: 2000
perplexity: 30

dataset:
  data_dir: /path/to/DeepShip_622
  split: validation
  data_type: wav_s

model:
  num_classes: 4
  student:
    p_encoder: 0.355
    p_classifier: 0.255

methods:
  - name: Baseline
    checkpoint: checkpoints/xxx/checkpoint_epoch_xxx.pth

  - name: Best
    checkpoint: checkpoints/xxx/best_student.pth
    checkpoint_key: student_state_dict

  - name: Logits KD
    checkpoint: checkpoints/xxx/best_student.pth
    checkpoint_key: student_state_dict

  - name: Feature KD
    checkpoint: checkpoints/xxx/best_student.pth
    checkpoint_key: student_state_dict
```

说明：

- `output_dir` 决定输出目录。如果命令行额外传入 `--output`，会覆盖 YAML 中的 `output_dir`。
- `checkpoint_key` 需要和 checkpoint 内部保存字段一致。蒸馏模型常用 `student_state_dict`，普通学生模型可能是 `model_state_dict` 或无需填写。
- 四个 checkpoint 都会加载到同一个 `AudioCfC` 学生网络结构中，用于比较学生表征空间。

## 默认绘图格式

当前配置默认只输出四张独立小图，不输出 2x2 合并图：

```yaml
plot:
  filename_suffix: single_column
  single_figsize: [1.65, 1.45]
  compact_layout: true
  title_fontsize: 9
  point_size: 3
  point_alpha: 0.85
  edge_linewidth: 0
  show_grid: false
  show_axis_labels: false
  hide_ticks: true
  show_legend: false
  save_comparison: false
  dpi: 600
```

其中：

- `single_figsize` 控制每张小图尺寸，单位为 inch。
- `save_comparison: false` 表示不生成合并图。
- `hide_ticks: true` 和 `show_axis_labels: false` 用于生成论文中更紧凑的子图。
- `filename_suffix` 会追加到输出文件名后，例如 `_single_column`。

## 第二步：运行

DeepShip：

```bash
python run_tsne_visualization.py --config configs/tsne_four_methods_deepship.yaml
```

ShipsEar：

```bash
python run_tsne_visualization.py --config configs/tsne_four_methods_shipsear.yaml
```

快速测试时可以减少样本数：

```bash
python run_tsne_visualization.py \
  --config configs/tsne_four_methods_deepship.yaml \
  --max_samples 500 \
  --perplexity 20
```

## 输出结果

默认输出到 YAML 中的 `output_dir`。以 ShipsEar 为例：

```text
results/tsne_four_methods_shipsear_v2/
```

默认会生成四张独立小图：

| 文件 | 说明 |
| --- | --- |
| `tsne_baseline_logits_single_column.png` | Baseline 小图 |
| `tsne_best_logits_single_column.png` | Best 小图 |
| `tsne_logits_kd_logits_single_column.png` | Logits KD 小图 |
| `tsne_feature_kd_logits_single_column.png` | Feature KD 小图 |
| `tsne_four_methods_logits_single_column.npz` | t-SNE 坐标和标签 |
| `tsne_run_summary_single_column.csv` | 本次运行的 checkpoint 和样本信息 |

注意：当前默认不会生成 `tsne_four_methods_logits_single_column.png` 这种合并图。

## LaTeX 2x2 示例

可以在论文中手动排版四张小图：

```latex
\begin{figure}[htbp]
    \centering
    \begin{minipage}{0.48\columnwidth}
        \centering
        \includegraphics[width=\linewidth]{figures/tsne_baseline_logits_single_column.png}
    \end{minipage}
    \begin{minipage}{0.48\columnwidth}
        \centering
        \includegraphics[width=\linewidth]{figures/tsne_best_logits_single_column.png}
    \end{minipage}

    \vspace{1mm}

    \begin{minipage}{0.48\columnwidth}
        \centering
        \includegraphics[width=\linewidth]{figures/tsne_logits_kd_logits_single_column.png}
    \end{minipage}
    \begin{minipage}{0.48\columnwidth}
        \centering
        \includegraphics[width=\linewidth]{figures/tsne_feature_kd_logits_single_column.png}
    \end{minipage}
    \caption{Four-method t-SNE visualization of logits.}
    \label{fig:tsne_four_methods}
\end{figure}
```

如果你使用双栏论文模板，并希望图片占单栏宽度，推荐使用 `\columnwidth`；如果希望跨双栏，则改用 `figure*` 和 `\textwidth`。

## 生成 2x2 合并图

如果后续仍想由脚本直接生成一张 2x2 合并图，把 YAML 中这一项改为：

```yaml
plot:
  save_comparison: true
```

此时会额外生成：

```text
tsne_four_methods_logits_single_column.png
```

## 可视化特征层

`feature_layer` 支持：

| 值 | 含义 | 建议用途 |
| --- | --- | --- |
| `logits` | 分类输出 | 论文主图推荐使用 |
| `pooled_seq` | 时序特征平均池化 | 查看较稳定的学生时序表征 |
| `seq_features` | 展平后的 CfC 序列特征 | 查看完整中间特征，维度较高 |
| `encoder` | CNN encoder 输出 | 查看前端声学编码特征 |

示例：

```bash
python run_tsne_visualization.py \
  --config configs/tsne_four_methods_deepship.yaml \
  --feature_layer pooled_seq
```

## 常见问题

### 为什么输出还是旧目录？

运行时脚本优先使用命令行 `--output`：

```text
--output > YAML output_dir > 默认 results/tsne_four_methods
```

如果你修改了 YAML 但输出仍在旧目录，检查运行命令中是否带了 `--output`，或者是否运行了另一份配置文件。

### 为什么没有合并图？

当前默认 `save_comparison: false`，只保存四张小图。需要合并图时改成 `true`。

### 如何调整小图尺寸？

修改：

```yaml
plot:
  single_figsize: [1.65, 1.45]
```

数值越大，单张图越大；LaTeX 中仍可通过 `\includegraphics[width=...]` 进一步缩放。
