# 四方法 t-SNE 分析快速使用

本项目现在保留一个主流程：对同一批验证样本，比较四个学生网络 checkpoint 的 t-SNE 分布。

四个方法建议对应为：

1. `Student Baseline`: 未蒸馏学生网络 baseline
2. `Proposed Distillation`: 你的蒸馏方法
3. `Logits KD`: 基于 logits 的蒸馏方法，例如 `kd`, `dkd`, `lskd`
4. `Feature KD`: 基于 feature 的蒸馏方法，例如 `rkd`, `pkt`, `sp`, `at`

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `batch_tsne_visualization.py` | 主脚本，读取四个 checkpoint，生成 2x2 对比图 |
| `run_tsne_visualization.py` | 快速启动脚本，也可以生成 YAML 模板 |
| `configs/tsne_four_methods_deepship.yaml` | DeepShip 四方法配置模板 |
| `configs/tsne_four_methods_shipsear.yaml` | ShipsEar 四方法配置模板 |

旧的多入口文档已删除，避免把“特征层对比”和“方法对比”混在一起。

## 第一步：修改 YAML

打开一个配置文件，例如：

```bash
configs/tsne_four_methods_deepship.yaml
```

重点修改：

```yaml
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
  - name: Student Baseline
    checkpoint: checkpoints/DeepShip/LNN_xxx/best_model.pth
    checkpoint_key: model_state_dict

  - name: Proposed Distillation
    checkpoint: checkpoints/DeepShip/proposed_xxx/best_student.pth
    checkpoint_key: student_state_dict

  - name: Logits KD
    checkpoint: checkpoints/DeepShip/comparison_distillation_kd/best_student.pth
    checkpoint_key: student_state_dict

  - name: Feature KD
    checkpoint: checkpoints/DeepShip/comparison_distillation_rkd/best_student.pth
    checkpoint_key: student_state_dict
```

说明：

- baseline 的 `best_model.pth` 通常使用 `checkpoint_key: model_state_dict`
- 蒸馏训练得到的 `best_student.pth` 通常使用 `checkpoint_key: student_state_dict`
- 四个 checkpoint 都会加载到同一个 `AudioCfC` 学生网络中，比较的是学生模型学到的表示

## 第二步：运行

DeepShip：

```bash
python run_tsne_visualization.py --config configs/tsne_four_methods_deepship.yaml
```

ShipsEar：

```bash
python run_tsne_visualization.py --config configs/tsne_four_methods_shipsear.yaml
```

也可以直接运行主脚本：

```bash
python batch_tsne_visualization.py \
  --config configs/tsne_four_methods_deepship.yaml \
  --feature_layer logits \
  --max_samples 2000 \
  --perplexity 30
```

## 可视化哪一层

`feature_layer` 支持：

| 值 | 含义 | 建议用途 |
| --- | --- | --- |
| `logits` | 分类输出 | 看最终分类空间，适合四方法主图 |
| `pooled_seq` | 时序特征平均池化 | 看更稳定的学生时序表示 |
| `seq_features` | 展平后的 CfC 序列特征 | 看完整中间特征，但维度更高 |
| `encoder` | CNN encoder 输出 | 看前端声学编码特征 |

论文里建议先用 `logits` 做主图；如果想强调 feature 蒸馏的作用，再补一张 `pooled_seq`。

## 输出结果

默认输出到 YAML 中的 `output_dir`，包含：

| 文件 | 说明 |
| --- | --- |
| `tsne_four_methods_<feature>.png` | 四方法 2x2 对比图 |
| `tsne_<method>_<feature>.png` | 每个方法的单独大图 |
| `tsne_four_methods_<feature>.npz` | t-SNE 坐标和标签，便于后续复画 |
| `tsne_run_summary.csv` | 本次运行使用的 checkpoint、特征层和样本数 |

## 常用命令

快速测试：

```bash
python run_tsne_visualization.py \
  --config configs/tsne_four_methods_deepship.yaml \
  --max_samples 500 \
  --perplexity 20
```

生成 feature 图：

```bash
python run_tsne_visualization.py \
  --config configs/tsne_four_methods_deepship.yaml \
  --feature_layer pooled_seq
```

生成新模板：

```bash
python run_tsne_visualization.py \
  --make-template \
  --config configs/my_tsne_four_methods.yaml
```
