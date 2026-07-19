# TMSKD overview 图张量形状与代码对应

本文档用于绘制 TMSKD overview/structure 图时标注每一步张量形状，并把论文符号与当前代码变量对应起来。

## 0. 全局符号

| 符号 | 含义 | 当前配置/代码默认值 | 代码位置 |
|---|---:|---:|---|
| `B` | batch size | 训练配置为 `16` | `configs/train_distillation_shipsear.yaml:3`, `configs/train_distillation_deepship.yaml:3` |
| `L` | 原始波形采样点数 | 论文按 3 s, 16 kHz 时为 `48000`；代码模型注释按 `(B,1,48000)` | `models/LNN.py:954` |
| `K` | 类别数 | ShipsEar: `5`; DeepShip: `4` | `configs/train_distillation_shipsear.yaml:34`, `configs/train_distillation_deepship.yaml:34` |
| `T_S` | 学生时间步数 | `16` | `configs/train_distillation_shipsear.yaml:42`, `models/LNN.py:916` |
| `C` | 学生时序特征维度 | `64` | `models/LNN.py:917`, `models/LNN.py:932` |
| `W` | BPCSCfC 窗口长度 | `4` | `models/LNN.py:889`, `models/LNN.py:934` |
| `N` | 窗口数 | `N=T_S/W=4` | `models/LNN.py:820` |
| `T_T` | 教师时间步数 | Wav2Vec2 当前约 `149` | `models/Audio_TeacherNet.py:43`, `models/Audio_TeacherNet.py:49` |
| `D_cnn` | 教师 CNN extractor 特征维度 | `512` | `models/Audio_TeacherNet.py:43` |
| `D_h` | 教师 transformer hidden 维度 | `1024` | `models/Audio_TeacherNet.py:49` |

## 1. 输入与数据流入口

| 图中模块 | 论文符号 | 代码变量 | 形状 | 代码位置 |
|---|---|---|---|---|
| Raw waveform for student | `s_S` | `audio_input[0]` / `inputs[0]` | `[B, 1, L]`，常用 `[B,1,48000]` | `datasets/audio_dataset.py:106-110`, `models/distillation.py:92` |
| Raw waveform for teacher | `s_T` | `audio_input[1]` / `inputs[1]` | `[B, 1, L]`；由 Wav2Vec2Processor 处理后进入教师 | `datasets/audio_dataset.py:112-117`, `models/distillation.py:109` |
| Batch input list | - | `inputs` | `[student_wav, teacher_wav]` | `train_distillation_shipsear.py:156-164`, `train_distillation_shipsear.py:224` |

图中建议标注：

```text
Raw waveform
student path: [B,1,L]
teacher path: [B,1,L]
```

## 2. 学生网络 AudioCfC

### 2.1 CNN Encoder

| 图中步骤 | 论文符号 | 代码变量 | 形状 | 代码位置 |
|---|---|---|---|---|
| 输入波形 | `s` | `x` | `[B,1,48000]` | `models/LNN.py:954` |
| 1D CNN encoder 输出 | `H_S` 或 `x_encoder` | `x_encoder = self.audio_encoder(x)` | `[B,64,16]` | `models/LNN.py:956` |
| 转置为时序格式 | `H_S` | `x_encoder = x_encoder.permute(0,2,1)` | `[B,16,64]` | `models/LNN.py:958` |
| encoder dropout 后 | `H_S` | `x_encoder = self.encoder_dropout(x_encoder)` | `[B,16,64]` | `models/LNN.py:959` |

CNN 逐层长度（输入 `L=48000` 时，按 PyTorch 默认 floor 计算）：

| 层 | 代码 | 输出形状 |
|---|---|---|
| BatchNorm1d | `BatchNorm1d(1)` | `[B,1,48000]` |
| Conv1d k7 s3 p3 | `Conv1d(1,32,...)` | `[B,32,16000]` |
| AvgPool1d(2) | `AvgPool1d(2)` | `[B,32,8000]` |
| Conv1d k5 s2 p2 | `Conv1d(32,64,...)` | `[B,64,4000]` |
| AvgPool1d(3) | `AvgPool1d(3)` | `[B,64,1333]` |
| Conv1d k3 s2 p1 | `Conv1d(64,64,...)` | `[B,64,667]` |
| AvgPool1d(4) | `AvgPool1d(4)` | `[B,64,166]` |
| Conv1d k3 s2 p1 | `Conv1d(64,64,...)` | `[B,64,83]` |
| AvgPool1d(5) | `AvgPool1d(5)` | `[B,64,16]` |
| Conv1d k3 s1 p1 | `Conv1d(64,64,...)` | `[B,64,16]` |

注意：`models/LNN.py` 中个别中文注释写成 `1334/167/84`，但按 PyTorch floor 公式，中间长度应为 `1333/166/83`；最终 `T_S=16` 不受影响。

图中建议标注：

```text
1D CNN Encoder
[B,1,L] -> H_S: [B,T_S,C] = [B,16,64]
```

### 2.2 BPCSCfC: 四路时序建模

入口：

| 图中步骤 | 论文符号 | 代码变量 | 形状 | 代码位置 |
|---|---|---|---|---|
| BPCSCfC 输入 | `H_S` | `x` | `[B,T_S,C] = [B,16,64]` | `models/LNN.py:858` |
| 翻转序列 | - | `x_flipped` | `[B,16,64]` | `models/LNN.py:859` |

Local branch：

| 图中步骤 | 论文符号 | 代码变量 | 形状 | 代码位置 |
|---|---|---|---|---|
| 按局部窗口 reshape | `X_l` | `x.view(B,N,W,C)` | `[B,4,4,64]` | `models/LNN.py:849` |
| 合并 batch 与窗口维 | `\tilde{X}_l` | `x_local` | `[B*N,W,C] = [4B,4,64]` | `models/LNN.py:849` |
| 前向局部 CfC | `F_local` | `out_local` / `fwd_local` | `[B,16,64]` | `models/LNN.py:850-851`, `models/LNN.py:860` |
| 反向局部 CfC | `B_local` | `bwd_local` | `[B,16,64]` | `models/LNN.py:863-864` |

Global branch：

| 图中步骤 | 论文符号 | 代码变量 | 形状 | 代码位置 |
|---|---|---|---|---|
| 跨窗口 reshape | `X_g` | `x.view(B,N,W,C).permute(0,2,1,3)` | `[B,W,N,C] = [B,4,4,64]` | `models/LNN.py:853` |
| 合并 batch 与窗口位置维 | `\tilde{X}_g` | `x_global` | `[B*W,N,C] = [4B,4,64]` | `models/LNN.py:853` |
| 前向全局 CfC | `F_global` | `out_global` / `fwd_global` | `[B,16,64]` | `models/LNN.py:854-855`, `models/LNN.py:860` |
| 反向全局 CfC | `B_global` | `bwd_global` | `[B,16,64]` | `models/LNN.py:863-865` |

融合：

| 图中步骤 | 论文符号 | 代码变量 | 形状 | 代码位置 |
|---|---|---|---|---|
| 四路 concat | `Concat(F_l,F_g,B_l,B_g)` | `merged` | full 模式下 `[B,16,256]` | `models/LNN.py:875` |
| Linear + LN + GELU | `F_S^t` | `fused` / `seq_features` | `[B,16,64]` | `models/LNN.py:877`, `models/LNN.py:960` |
| 返回四路分支 | - | `fl, fg, bl, bg` | 每个 `[B,16,64]` | `models/LNN.py:879`, `models/LNN.py:960` |

图中建议标注：

```text
BPCSCfC
local:  [B,16,64] -> [B*N,W,64] -> [B,16,64]
global: [B,16,64] -> [B*W,N,64] -> [B,16,64]
concat four streams: [B,16,256]
F_S^t: [B,16,64]
```

### 2.3 DRASP 与学生最终分类

| 图中步骤 | 论文符号 | 代码变量 | 形状 | 代码位置 |
|---|---|---|---|---|
| DRASP 输入转置 | `(F_S^t)^T` | `seq_features.permute(0,2,1)` | `[B,64,16]` | `models/LNN.py:961` |
| Global ASP | `F_g` | `global_stats` | `[B,2C] = [B,128]` | `models/LNN.py:793`, `models/LNN.py:764-770` |
| Local segmentation | - | `x_segmented` | segment_len=4 时 `[B*4,64,4]` | `models/LNN.py:799` |
| Local ASP for each segment | - | `local_stats_all` | `[B*4,128]` | `models/LNN.py:802` |
| Restore segments | - | `local_stats_view` | `[B,4,128]` | `models/LNN.py:804` |
| Max over segments | `F_l` | `local_stats_max` | `[B,128]` | `models/LNN.py:806` |
| Global + Local concat | `F_pooled` | `pooled_features` | `[B,4C] = [B,256]` | `models/LNN.py:809`, `models/LNN.py:961` |
| Classifier | `Z_S` | `out` / `student_logits` | `[B,K]` | `models/LNN.py:949-952`, `models/LNN.py:962-963` |

图中建议标注：

```text
DRASP
Global ASP: [B,64,16] -> [B,128]
Local ASP:  [B,64,16] -> [B,4,128] -> [B,128]
Concat: [B,256] -> Classifier -> Z_S: [B,K]
```

## 3. 学生逐时间步 logits

注意：`AudioCfC.forward()` 返回的第二个变量 `seq_features` 是时序特征 `[B,16,64]`，不是 logits。逐时间步 logits 在蒸馏包装模型中额外线性投影得到。

| 图中步骤 | 论文符号 | 代码变量 | 形状 | 代码位置 |
|---|---|---|---|---|
| 学生时序特征 | `F_S^t` | `stu_sequence_logits` before projection / `seq_features` | `[B,16,64]` | `models/distillation.py:92` |
| 线性投影到类别维 | `Z_S^t` | `stu_sequence_logits = self.stu_linear(stu_sequence_logits)` | `[B,16,K]` | `models/distillation.py:63`, `models/distillation.py:94` |
| 学生最终 logits | `Z_S` | `student_logits` | `[B,K]` | `models/distillation.py:92`, `models/distillation.py:112` |

图中建议标注：

```text
Temporal classifier / stu_linear
F_S^t: [B,16,64] -> Z_S^t: [B,16,K]
```

## 4. 教师网络 Wav2Vec2

| 图中步骤 | 论文符号 | 代码变量 | 形状 | 代码位置 |
|---|---|---|---|---|
| 教师输入 squeeze | `s_T` | `x = torch.squeeze(x, dim=1)` | `[B,L]` | `models/Audio_TeacherNet.py:35` |
| Wav2Vec2 forward | - | `wav2vec_result` | HuggingFace output | `models/Audio_TeacherNet.py:40` |
| CNN extractor features | `H_T^{cnn}` | `output_extract_features` / `output_cnn_features` | `[B,T_T,512]`，常见 `[B,149,512]` | `models/Audio_TeacherNet.py:43-46`, `models/distillation.py:109` |
| Last hidden state | `H_T` | `output_last_hidden_state` | `[B,T_T,1024]`，常见 `[B,149,1024]` | `models/Audio_TeacherNet.py:49-51` |
| Mean pooling | - | `torch.mean(x_last_hidden_state, dim=1)` | `[B,1024]` | `models/Audio_TeacherNet.py:62` |
| Teacher classifier | `Z_T` | `x` / `teacher_logits` | `[B,K]` | `models/Audio_TeacherNet.py:22-26`, `models/Audio_TeacherNet.py:63-65` |

图中建议标注：

```text
Frozen Wav2Vec2 Teacher
H_T^{cnn}: [B,149,512]
H_T:       [B,149,1024]
Z_T:       [B,K]
```

## 5. MemKD 分支

当前实现的 MemKD 使用学生 CNN encoder 后的 `x_encoder`，而不是 BPCSCfC 后的 `seq_features`。两者形状都是 `[B,16,64]`，但语义不同：代码里对齐的是 `CNN Encoder` 后的学生特征与教师 Wav2Vec2 CNN extractor 特征。

| 图中步骤 | 论文符号 | 代码变量 | 形状 | 代码位置 |
|---|---|---|---|---|
| 学生特征输入 | `H_S` | `x_encoder` / `student_h` | `[B,16,64]` | `models/distillation.py:112`, `utils/distillation_loss.py:113` |
| 教师特征输入 | `H_T^{cnn}` | `output_cnn_features` / `teacher_h` | `[B,149,512]` | `models/distillation.py:109`, `utils/distillation_loss.py:113` |
| 教师转置 | - | `t_h = teacher_h.permute(0,2,1)` | `[B,512,149]` | `utils/distillation_loss.py:119` |
| 时间插值对齐 | `Interp(H_T,T_S)` | `F.interpolate(..., size=student_h.size(1))` | `[B,512,16]` | `utils/distillation_loss.py:120` |
| 转回时序格式 | `\hat{H}_T` | `t_h = t_h.permute(0,2,1)` | `[B,16,512]` | `utils/distillation_loss.py:121` |
| 学生时序特征 | `H_S` | `s_h` | `[B,16,64]` | `utils/distillation_loss.py:123` |
| 当前状态 | `H^t` | `h_t_teacher`, `h_t_student` | teacher `[B,16-z,512]`; student `[B,16-z,64]` | `utils/distillation_loss.py:130-134` |
| 偏移状态 | `H^{t+z}` | `h_tz_teacher`, `h_tz_student` | teacher `[B,16-z,512]`; student `[B,16-z,64]` | `utils/distillation_loss.py:131-135` |
| 状态差分 | `\Delta H^t` | `delta_teacher`, `delta_student` | teacher `[B,16-z,512]`; student `[B,16-z,64]` | `utils/distillation_loss.py:138-139` |
| 归一化差分 | `F_t` | `f_t`, `f_s` | teacher `[B,16-z,512]`; student `[B,16-z,64]` | `utils/distillation_loss.py:143-148` |
| 幅度轨迹 | `M_t` | `mag_t`, `mag_s` | `[B,16-z]` | `utils/distillation_loss.py:151-152` |
| Smooth L1 | `L_MemKD(z)` | `loss` | scalar | `utils/distillation_loss.py:155-156` |
| 短期损失 | `L_s` | `memkd_loss_short` | scalar, `z=1` | `utils/distillation_loss.py:363` |
| 长期损失 | `L_l` | `memkd_loss_long` | scalar, `z=random` | `utils/distillation_loss.py:365-366` |
| MemKD 总损失 | `L_M` | `memkd_loss` | scalar | `utils/distillation_loss.py:367` |

当前配置：`z_random_range='half'`，因此长期偏移 `z` 从 `[2, T_S/2]` 采样；`T_S=16` 时约为 `[2,8]`。对应代码：`utils/distillation_loss.py:160-179`。

图中建议标注：

```text
MemKD
Student H_S: [B,16,64]
Teacher H_T^{cnn}: [B,149,512] -> Interp -> [B,16,512]
Delta over z: [B,16-z,*]
Magnitude trajectory: [B,16-z]
L_M = 0.5 L_s(z=1) + 1.0 L_l(z~U(2,8))
```

## 6. TS-T / KL_Temp 分支

代码中论文 TS-T 对应 `distill_type='MTSKD_Temp'` 下的 `KL_Temp` 分支：教师最终 logits 先做 Z-score 标准化，再沿时间维广播，与学生逐时间步 logits 对齐。

| 图中步骤 | 论文符号 | 代码变量 | 形状 | 代码位置 |
|---|---|---|---|---|
| 教师最终 logits | `Z_T` | `teacher_logits` | `[B,K]` | `models/distillation.py:109`, `utils/distillation_loss.py:370` |
| 学生逐时间步 logits | `Z_S^t` | `stu_seq_logits` | `[B,16,K]` | `models/distillation.py:94`, `utils/distillation_loss.py:371` |
| 学生最终 logits | `Z_S` | `student_logits` | `[B,K]` | `models/distillation.py:92`, `utils/distillation_loss.py:385` |
| Z-score 标准化教师 logits | `\hat{Z}_T` | `std_teacher_logits` | `[B,K]` | `utils/distillation_loss.py:185-189`, `utils/distillation_loss.py:370` |
| Z-score 标准化学生序列 logits | `\hat{Z}_S^t` | `std_stu_seq_logits` | `[B,16,K]` | `utils/distillation_loss.py:371` |
| 教师 soft label | `softmax(\hat{Z}_T)` | `soft_teacher` | `[B,K]` | `utils/distillation_loss.py:373` |
| 时间维广播 | `Z_T^t` | `soft_teacher_expanded` | `[B,16,K]` | `utils/distillation_loss.py:374` |
| 学生 log prob | `log_softmax(\hat{Z}_S^t)` | `soft_student_seq` | `[B,16,K]` | `utils/distillation_loss.py:375` |
| 每时间步 KL | `D_KL(Z_T^t || Z_S^t)` | `kl_per_step` | `[B,16]` | `utils/distillation_loss.py:378-380` |
| 时间加权 | - | `weighted_kl` | `[B]` | `utils/distillation_loss.py:381-382` |
| 序列蒸馏损失 | `L_TS-T^seq` | `seq_loss` | scalar | `utils/distillation_loss.py:383` |
| 最终 logits KL | `L_TS-T^final` | `final_loss` | scalar | `utils/distillation_loss.py:385-388` |
| TS-T 总损失 | `L_TS-T` | `kl_loss` | scalar | `utils/distillation_loss.py:389` |

图中建议标注：

```text
TS-T / KL_Temp
Z_T: [B,K] -> broadcast -> [B,16,K]
Z_S^t: [B,16,K]
Z_S: [B,K]
Z-score std = adaptive temperature
L_TS-T = beta L_seq + (1-beta) L_final
```

## 7. 总训练损失

| 图中步骤 | 论文符号 | 代码变量 | 形状 | 代码位置 |
|---|---|---|---|---|
| 硬标签 CE | `L_CE` | `hard_loss` | scalar | `utils/distillation_loss.py:199` |
| MemKD 损失 | `L_M` | `memkd_loss` | scalar | `utils/distillation_loss.py:367` |
| TS-T 损失 | `L_TS-T` | `kl_loss` | scalar | `utils/distillation_loss.py:389` |
| MemKD/TS-T 融合 | `lambda_m L_M + (1-lambda_m)L_TS-T` | `soft_loss, memkd_weight = self._mix_mtskd_loss(...)` | scalar | `utils/distillation_loss.py:391`, `utils/distillation_loss.py:181-183` |
| 总损失 | `L_total` | `total_loss = hard_loss + kd_weight * soft_loss` | scalar | `utils/distillation_loss.py:409` |

注意：论文公式写为

```text
L_total = L_CE + lambda_m L_M + (1-lambda_m) L_TS-T
```

代码中还支持 `kd_weight` 动态蒸馏权重；当前配置 `USE_DYNAMIC_DISTILL_WEIGHT: false`，所以 `kd_weight=1.0`，与论文公式一致。对应代码：`utils/distillation_loss.py:395-406`。

## 8. overview 图最终推荐标注版本

可以直接把下面这些标注贴到图中：

```text
Raw waveform:
  s_S, s_T: [B,1,L] (L=48000)

Student LNN:
  CNN Encoder: [B,1,L] -> H_S / x_encoder: [B,16,64]
  BPCSCfC:
    local:  [B,16,64] -> [B*N,W,64] -> [B,16,64]
    global: [B,16,64] -> [B*W,N,64] -> [B,16,64]
    concat four streams: [B,16,256] -> F_S^t: [B,16,64]
  Temporal classifier:
    F_S^t: [B,16,64] -> Z_S^t: [B,16,K]
  DRASP:
    [B,64,16] -> Global [B,128] + Local [B,128] -> [B,256]
  Classifier:
    [B,256] -> Z_S: [B,K]

Teacher Wav2Vec2:
  input: [B,1,L] -> squeeze: [B,L]
  extractor feature H_T^{cnn}: [B,149,512]
  last hidden H_T: [B,149,1024]
  mean pooling + classifier -> Z_T: [B,K]

MemKD:
  H_S / x_encoder: [B,16,64]
  H_T^{cnn}: [B,149,512] -> Interp -> [B,16,512]
  Delta_z + norm + magnitude: [B,16-z]
  L_M = 0.5 L_s(z=1) + 1.0 L_l(z~U(2,8))

TS-T:
  Z_T: [B,K] -> broadcast -> [B,16,K]
  Z_S^t: [B,16,K]
  Z_S: [B,K]
  Z-score adaptive temperature + KL
  L_TS-T = beta L_seq + (1-beta) L_final

Loss:
  L_total = L_CE + lambda_m L_M + (1-lambda_m) L_TS-T
```

## 9. 论文符号与代码变量速查

| 论文符号 | 代码变量 | 形状 |
|---|---|---|
| `s` | `audio_input[0]`, `audio_input[1]` | `[B,1,L]` |
| `H_S` | `x_encoder` | `[B,16,64]` |
| `F_S^t` | `seq_features` | `[B,16,64]` |
| `Z_S^t` | `stu_seq_logits` | `[B,16,K]` |
| `Z_S` | `student_logits` | `[B,K]` |
| `H_T^{cnn}` | `output_cnn_features` | `[B,149,512]` |
| `H_T` | `output_last_hidden_state`，在包装模型中变量名为 `teacher_all_hidden_states` | `[B,149,1024]` |
| `Z_T` | `teacher_logits` | `[B,K]` |
| `F_local` | `fl` / `fwd_local` | `[B,16,64]` |
| `F_global` | `fg` / `fwd_global` | `[B,16,64]` |
| `B_local` | `bl` / `bwd_local` | `[B,16,64]` |
| `B_global` | `bg` / `bwd_global` | `[B,16,64]` |

