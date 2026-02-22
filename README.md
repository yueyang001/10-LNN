**MTSKD_Temp 流程图**

下面展示 `distillation_loss.DistillationLoss` 中 `distill_type='MTSKD_Temp'` 的简要流程图（Mermaid）：

```mermaid
graph LR
	A[Start: forward()] --> B[Inputs\n(student_logits, stu_sequence_logits, x_encoder, teacher_logits, output_cnn_features, labels)]
	B --> C{MemKD_Temp Part}
	B --> D{KL_Temp Part}

	subgraph MemKD
		C1[Align teacher features: interpolate 149->16]
		C2[Compute delta: Δh = h(t+z) - h(t)]
		C3[Normalize: f = Δh / ||h(t)||]
		C4[Magnitude: mag = ||f||]
		C5[Loss: SmoothL1(mag_student, mag_teacher) for z=1 and z=random]
		C --> C1 --> C2 --> C3 --> C4 --> C5 --> E1[memkd_loss]
	end

	subgraph KL_Temp
		D1[Logit standardization: z-score teacher & student seq]
		D2[Expand teacher logits to time dim]
		D3[Compute per-step KL: KL(log_softmax(std_student_seq), softmax(std_teacher_expanded))]
		D4[Weight by time_weights and mean -> seq_loss]
		D5[final_loss = _compute_kl_loss(std(student_logits), std(teacher_logits))]
		D --> D1 --> D2 --> D3 --> D4 --> D5 --> E2[kl_loss = beta*seq_loss + (1-beta)*final_loss]
	end

	E1 --> F[Combine MemKD & KL]
	E2 --> F
	F --> G[memkd_weight = sigmoid(mtskd_weight) ; kl_weight = 1 - sigmoid(mtskd_weight)]
	G --> H[soft_loss = memkd_weight * memkd_loss + kl_weight * kl_loss]
	H --> I[kd_weight (dynamic by epoch) ; hard_loss = CrossEntropy(student_logits, labels)]
	I --> J[total_loss = hard_loss + kd_weight * soft_loss]
	J --> K[Return: total_loss, hard_loss, soft_loss, alpha, beta, memkd_weight]
	style MemKD fill:#f9f,stroke:#333,stroke-width:1px
	style KL_Temp fill:#9ff,stroke:#333,stroke-width:1px
```

简短说明：
- MemKD 部分通过对齐教师特征并计算时间差分的幅度差异得到 `memkd_loss`。
- KL_Temp 部分通过对 logits 做 Z-score 标准化后计算序列级和最终输出级的 KL，按 `beta` 混合得到 `kl_loss`。
- 最终用可学习的 `mtskd_weight`（经 sigmoid）在两者间加权，得到 `soft_loss`，并与交叉熵 `hard_loss` 按 `kd_weight(epoch)` 组合为 `total_loss`。

Reference: see `utils/distillation_loss.py` for implementation details.

