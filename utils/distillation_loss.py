from sympy import beta
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class DistillationLoss(nn.Module):
    """知识蒸馏损失函数 - 支持 KL 和 MSE 两种模式"""
    def __init__(self, temperature=4.0, alpha=0.5, learnable_alpha=True, 
                 seq_len=16, weight_type='uniform', distill_type='base', num_classes=4):
        """
        Args:
            distill_type: 'kl' 或 'mse'或 'MemKD' 或 'MTSKD' 或 'MTSKD_Temp'
        """
        super().__init__()
        self.temperature = temperature
        self.distill_type = distill_type
        
        if learnable_alpha: # 可学习的alpha
            self.alpha = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer('alpha', torch.tensor(alpha))
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

        self.beta = nn.Parameter(torch.tensor(0.0)) # 0.5
        
        # MTSKD 可学习权重 (只在MTSKD模式时使用)
        # self.mtskd_weight = nn.Parameter(torch.tensor(0.0))  # MemKD权重，KL权重为(1-mtskd_weight) 0.5
        # self.mtskd_weight = nn.Parameter(torch.tensor(-1.0))  # sigmoid ≈ 0.27
        self.mtskd_weight = nn.Parameter(torch.tensor(0.0))

        
        weights = self._create_weights(seq_len, weight_type) # weight: torch.Size([16])
        # print(f"weight: {weights.shape}")
        self.register_buffer('time_weights', weights)
        self.z_long = 20
        # 必须定义这个 Projector 来对齐 LNN(64) 和 Wav2Vec2(1024)
        self.projector = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.LayerNorm(1024) # 增加稳定性
        )
        self.seq_proj = nn.Linear(64, num_classes)
        self.feat_projector = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 5)
        )
        self.current_epoch = 0
        self.total_epoch = 100  # 默认，可从外部传
    def set_epoch(self, epoch, total_epoch):
        self.current_epoch = epoch
        self.total_epoch = total_epoch

    def _create_weights(self, seq_len, weight_type):
        if weight_type == 'linear':
            weights = torch.arange(1, seq_len + 1, dtype=torch.float32)
        elif weight_type == 'exponential':
            weights = torch.exp(torch.arange(seq_len, dtype=torch.float32) / seq_len * 3)
        elif weight_type == 'last_only':
            weights = torch.zeros(seq_len)
            weights[-10:] = 1.0
        elif weight_type == 'uniform':
            weights = torch.ones(seq_len)
        elif weight_type == 'zero':
            weights = torch.zeros(seq_len)
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")
        
        weights = weights / weights.sum()
        return weights
    
    def _compute_kl_loss(self, student_logits, teacher_logits):
        """
        应用 Logit Standardization (Norm-KD) 后的 KL 散度计算
        注意：此时不再使用 self.temperature
        """
        # 1. 对教师和学生 Logit 分别进行标准化 (Z-score)
        # dim=-1 表示在类别（num_classes）维度上计算
        def logit_std(x):
            return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-5)

        std_student = logit_std(student_logits)
        std_teacher = logit_std(teacher_logits)

        # 2. 计算 Softmax (标准化过程已隐含了自适应温度控制)
        soft_teacher = F.softmax(std_teacher, dim=-1)
        soft_student = F.log_softmax(std_student, dim=-1)

        # 3. 计算 KL 散度
        kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')

        # 4. 关键点：不再乘 (self.temperature ** 2)
        # 论文实验表明，标准化后量级已对齐，通常直接返回 kl_loss 即可
        return kl_loss
    
    def _compute_mse_loss(self, student_logits, teacher_logits):
        """计算 MSE 损失"""
        # 直接对 logits 计算 MSE，不需要 softmax
        return self.mse_loss(student_logits, teacher_logits)
    
    def _compute_memkd_loss(self, student_h, teacher_h, z):
        """
        student_h: [B, 16, 64]
        teacher_h: [B, 149, 512]
        z: 偏移量
        """
        # ---- 步骤 A: 时间步对齐 (教师 149 -> 16) ----
        # interpolate 期望 [B, C, L], 所以先转置
        t_h = teacher_h.permute(0, 2, 1) # [B, 512, 149]
        t_h = F.interpolate(t_h, size=student_h.size(1), mode='linear', align_corners=True)
        t_h = t_h.permute(0, 2, 1) # [B, 16, 512]
        
        s_h = student_h # [B, 16, 64]
        
        # ---- 步骤 B: 计算内存差异 (Memory Discrepancy) ----
        # 确保 t+z 不超过序列长度 16
        T = s_h.size(1)
        if z >= T: z = T // 4 # 防止 z 过大
        
        # 提取 t 时刻和 t+z 时刻的状态
        h_t_teacher = t_h[:, :T-z, :]
        h_tz_teacher = t_h[:, z:, :]
        
        h_t_student = s_h[:, :T-z, :]
        h_tz_student = s_h[:, z:, :]
        
        # 计算位移 Δh = h(t+z) - h(t)
        delta_teacher = h_tz_teacher - h_t_teacher
        delta_student = h_tz_student - h_t_student
        
        # ---- 步骤 C: 归一化 (公式 4 & 5) ----
        eps = 1e-8
        # 教师归一化映射 f_t
        norm_t = torch.norm(h_t_teacher, p=2, dim=-1, keepdim=True) + eps
        f_t = delta_teacher / norm_t
        
        # 学生归一化映射 f_s
        norm_s = torch.norm(h_t_student, p=2, dim=-1, keepdim=True) + eps
        f_s = delta_student / norm_s
        
        # ---- 步骤 D: 计算幅度差异 (跨维度对齐) ----
        # 计算向量的 L2 范数（幅度），使 512 维和 64 维变为可比的标量轨迹
        mag_t = torch.norm(f_t, p=2, dim=-1)
        mag_s = torch.norm(f_s, p=2, dim=-1)

        # 使用 Smooth L1 Loss (论文推荐)
        loss = F.smooth_l1_loss(mag_s, mag_t)
        
        return loss
    
    def _logit_standardization(self, logits, dim=-1):
        """
        Logit Standardization (Z-score) 预处理
        公式: (x - mean) / std
        """
        mean = logits.mean(dim=dim, keepdim=True)
        std = logits.std(dim=dim, keepdim=True) + 1e-5  # 防止除以 0
        return (logits - mean) / std

    def forward(self, student_logits, stu_sequence_logits, fl, fg, bl, bg, x_encoder, teacher_logits, output_cnn_features,labels, teacher_all_hidden_states=None):
        _, seq_len, _ = stu_sequence_logits.shape
    
        # 硬标签损失
        hard_loss = self.ce_loss(student_logits, labels)
        
        if self.distill_type == 'kl':
            # ============ 时序 KL 散度方法 ================
            # [新增这一行] 给 memkd_weight 一个默认值，防止 return 报错
            memkd_weight = torch.tensor(0.0).to(stu_sequence_logits.device)
            """
            stu_sequence_logits:[batch,seq_len,num_classes] [B, 16, 5]
            teacher_logits: [batch, num_classes] 
            """
            # ============ 应用 Logit Standardization ============
            # 对教师 Logit 进行标准化
            std_teacher_logits = self._logit_standardization(teacher_logits) # [B, C]
            
            # 对学生时序 Logit 进行标准化 (在类别维度上)
            std_stu_seq_logits = self._logit_standardization(stu_sequence_logits) # [B, T, C]

            # 计算 Softmax (此时不再需要除以固定的 self.temperature)
            soft_teacher = F.softmax(std_teacher_logits, dim=-1) # -> [B, C]
            soft_teacher_expanded = soft_teacher.unsqueeze(1).expand(-1, seq_len, -1) # -> [B, T, C]
            
            # 学生端使用 log_softmax
            soft_student_seq = F.log_softmax(std_stu_seq_logits, dim=-1) # -> [B, T, C]

            # ===== time weights =====
            time_weights_truncated = self.time_weights[:seq_len]  # [T] 

            # ============ 计算 KL 散度 ============
            kl_per_step = F.kl_div(
                soft_student_seq, soft_teacher_expanded, reduction='none' 
            ).sum(dim=-1) # [B, T]
            
            weighted_kl = (kl_per_step * time_weights_truncated).sum(dim=-1)
            # 注意：使用标准化后，论文建议不再乘以 temperature^2，或直接设为 1
            seq_loss = weighted_kl.mean() 
            
            # 同样对最终输出 final_loss 应用标准化 
            final_loss = self._compute_kl_loss(
                self._logit_standardization(student_logits), 
                std_teacher_logits
            )
            
            soft_loss = self.beta * seq_loss + (1 - self.beta) * final_loss
        
        elif self.distill_type == 'Tser':
        # ============ 特征对齐与时序 KL 蒸馏 ================
            # [新增这一行] 给 memkd_weight 一个默认值，防止 return 报错
            memkd_weight = torch.tensor(0.0).to(stu_sequence_logits.device)
            # 教师特征 teacher_all_hidden_states: [B, 149, 1024]
            # 学生序列 stu_sequence_logits: [B, 16, 5]

            # 1. 线性投影：将教师的 1024 维降至 5 维
            # [B, 149, 1024] -> [B, 149, 5]
            projected_teacher = self.feat_projector(teacher_all_hidden_states)

            # 2. 时间维度对齐：将 149 帧对齐到 16 帧
            # 使用 interpolate 需要将通道维提到前面：[B, 5, 149]
            projected_teacher = projected_teacher.transpose(1, 2) 
            
            # 缩放到学生的序列长度 (16)
            # [B, 5, 149] -> [B, 5, 16]
            aligned_teacher_feat = F.interpolate(
                projected_teacher, 
                size=stu_sequence_logits.shape[1], 
                mode='linear', 
                align_corners=False
            )
            
            # 转置回 [B, 16, 5]
            aligned_teacher_feat = aligned_teacher_feat.transpose(1, 2)

            # 3. 计算软标签 (Softmax)
            # 注意：教师特征经过线性层后是原始数值，需要做 Softmax 变成概率分布
            soft_teacher = F.softmax(aligned_teacher_feat / self.temperature, dim=-1) # [B, 16, 5]
            soft_student_seq = F.log_softmax(stu_sequence_logits / self.temperature, dim=-1) # [B, 16, 5]

            # 4. 计算 KL 散度
            kl_per_step = F.kl_div(
                soft_student_seq, 
                soft_teacher, 
                reduction='none'
            ).sum(dim=-1) # [B, 16]
            # ============ 关键修复点 ============
            # 获取当前实际的序列长度（学生网络的 T）
            actual_seq_len = kl_per_step.size(1) 
            
            # 确保 time_weights 被正确截取并移动到相同的设备（GPU/CPU）
            # 假设 self.time_weights 是在 __init__ 中定义的预设权重
            time_weights_truncated = self.time_weights[:actual_seq_len].to(kl_per_step.device)
            # ==================================
            # 5. 权重计算
            weighted_kl = (kl_per_step * time_weights_truncated).sum(dim=-1)
            seq_loss = weighted_kl.mean() * (self.temperature ** 2)

            # 最终输出的 KL (保持原样，用于全局对齐)
            final_loss = self._compute_kl_loss(student_logits, teacher_logits)
            soft_loss = self.beta * seq_loss + (1 - self.beta) * final_loss
            
        elif self.distill_type == 'MemKD':
            # ============ MemKD 核心实现 ============
            # print("MemKD todo") # for test
            """
            学生特征:x_encoder: [batch,time,features] [B, 16, 64]
            教师特征:output_cnn_features:[batch,seq_len,hidden_size] [B, 149, 512] 
            """
            # 1. 计算短程内存差异损失 (z=1)
            loss_short = self._compute_memkd_loss(x_encoder, output_cnn_features, z=1)
            
            # 2. 计算远程内存差异损失 (z > 1，例如 z=4 或随机)
            T_max = x_encoder.size(1) # 16
            z_random = random.randint(2, T_max // 2) # 在 2 到 8 之间随机选一个跨度
            
            loss_long = self._compute_memkd_loss(x_encoder, output_cnn_features, z=z_random)
            
            # 3. 按照公式 (7) 组合损失
            # alpha 默认为 1.0 (CE Loss 在外部计算), 这里计算 L_KD 项
            # 这里的 0.5 和 1.0 是短/长程的内部权重，soft_loss 即为最终的 L_MemKD
            soft_loss = 0.5 * loss_short + 1.0 * loss_long

        elif self.distill_type == 'mse':
            # ============ MSE 方法 ================
            teacher_expanded = teacher_logits.unsqueeze(1).expand(-1, seq_len, -1)
            
            # 每个时间步的 MSE [B, seq_len]
            mse_per_step = ((stu_sequence_logits - teacher_expanded) ** 2).mean(dim=-1)
            
            # 加权求和
            weighted_mse = (mse_per_step * self.time_weights).sum(dim=-1)
            seq_loss = weighted_mse.mean()
            
            # 最终输出的 MSE
            final_loss = self._compute_mse_loss(student_logits, teacher_logits)
            soft_loss = self.beta * seq_loss + (1 - self.beta) * final_loss
        elif self.distill_type == 'base':
            """
            teacher_logits: [B, num_classes]
            student_logits: [B, seq_len, num_classes]
            """
            soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
            soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
            soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
            soft_loss = soft_loss * (self.temperature ** 2)
        elif self.distill_type == 'MTSKD':
            # ============ MTSKD (MemKD + KL) 混合蒸馏方法 ============
            """
            MTSKD 方法结合:
            1. MemKD: 通过内存差异学习细粒度时序特征
            2. KL: 通过分布匹配学习类别间关系
            综合这两个损失可以更全面地指导学生网络学习
            """
            
            # ---- 方法1: MemKD 部分 ----
            # 1. 计算短程内存差异损失 (z=1)
            memkd_loss_short = self._compute_memkd_loss(x_encoder, output_cnn_features, z=1)
            
            # 2. 计算远程内存差异损失 (z > 1)
            T_max = x_encoder.size(1) # 16
            z_random = random.randint(2, T_max // 2) # 在 2 到 8 之间随机选一个跨度
            memkd_loss_long = self._compute_memkd_loss(x_encoder, output_cnn_features, z=z_random)
            
            # MemKD 损失 (与原始MemKD相同的权重)
            memkd_loss = 0.5 * memkd_loss_short + 1.0 * memkd_loss_long
            
            # ---- 方法2: KL 部分 ----
            # 时序 KL 散度方法
            soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
            soft_teacher_expanded = soft_teacher.unsqueeze(1).expand(-1, seq_len, -1)
            soft_student_seq = F.log_softmax(stu_sequence_logits / self.temperature, dim=-1)
            
            kl_per_step = F.kl_div(
                soft_student_seq, soft_teacher_expanded, reduction='none'
            ).sum(dim=-1)
            
            # 确保权重与实际序列长度匹配
            batch_size, actual_seq_len = kl_per_step.shape
            time_weights_truncated = self.time_weights[:actual_seq_len]
            
            weighted_kl = (kl_per_step * time_weights_truncated).sum(dim=-1)
            seq_loss = weighted_kl.mean() * (self.temperature ** 2)
            
            # 最终输出的 KL
            final_loss = self._compute_kl_loss(student_logits, teacher_logits)
            kl_loss = self.beta * seq_loss + (1 - self.beta) * final_loss
            
            # ---- MTSKD 融合 ----
            # 将 MemKD 和 KL 损失加权组合
            
            # 方案选择：
            USE_LEARNABLE_WEIGHT = True  # 可以设置为True体验可学习权重
            
            if USE_LEARNABLE_WEIGHT:
                # 方案1：使用可学习权重
                # memkd_weight = torch.sigmoid(self.mtskd_weight)
                # kl_weight = 1.0 - self.memkd_weight
                # soft_loss = self.memkd_weight * memkd_loss + kl_weight * kl_loss
                soft_loss = self.mtskd_weight * memkd_loss + (1 - self.mtskd_weight) * kl_loss
                # print(f"MTSKD Learnable - MemKD loss: {memkd_loss:.4f}, KL loss: {kl_loss:.4f}")
                # print(f"MTSKD Learnable - MemKD weight: {memkd_weight:.4f}, KL weight: {kl_weight:.4f}")
                # print(f"MTSKD Learnable - Combined: {soft_loss:.4f}")
            else:
                # 方案2：使用固定权重（推荐，避免过多的可学习参数）
                soft_loss = 0.5 * memkd_loss + 0.5 * kl_loss
                # print(f"MTSKD Fixed - MemKD loss: {memkd_loss:.4f}, KL loss: {kl_loss:.4f}, Combined: {soft_loss:.4f}")
        elif self.distill_type == 'MTSKD_Temp':
            # ============ MTSKD_Temp (MemKD + KL_Temp) 混合蒸馏方法 ============
            """
            MTSKD_Temp 方法结合:
            1. MemKD: 通过内存差异学习细粒度时序特征
            2. KL_Temp: 通过分布匹配学习类别间关系
            综合这两个损失可以更全面地指导学生网络学习
            """
            
            # ---- 方法1: MemKD 部分 ----
            # 1. 计算短程内存差异损失 (z=1)
            memkd_loss_short = self._compute_memkd_loss(x_encoder, output_cnn_features, z=1)
            
            # 2. 计算远程内存差异损失 (z > 1)
            T_max = x_encoder.size(1) # 16
            z_random = random.randint(2, T_max // 2) # 在 2 到 8 之间随机选一个跨度
            memkd_loss_long = self._compute_memkd_loss(x_encoder, output_cnn_features, z=z_random)
            
            # MemKD 损失 (与原始MemKD相同的权重)
            memkd_loss = 0.5 * memkd_loss_short + 1.0 * memkd_loss_long
            
            # ---- 方法2: KL_Temp 部分 ----
            # ============ 应用 Logit Standardization ============
            # 对教师 Logit 进行标准化
            std_teacher_logits = self._logit_standardization(teacher_logits) # [B, C]
            
            # 对学生时序 Logit 进行标准化 (在类别维度上)
            std_stu_seq_logits = self._logit_standardization(stu_sequence_logits) # [B, T, C]

            # 计算 Softmax (此时不再需要除以固定的 self.temperature)
            soft_teacher = F.softmax(std_teacher_logits, dim=-1) # -> [B, C]
            soft_teacher_expanded = soft_teacher.unsqueeze(1).expand(-1, seq_len, -1) # -> [B, T, C]
            
            # 学生端使用 log_softmax
            soft_student_seq = F.log_softmax(std_stu_seq_logits, dim=-1) # -> [B, T, C]

            # ===== time weights =====
            time_weights_truncated = self.time_weights[:seq_len]  # [T] 

            # ============ 计算 KL 散度 ============
            kl_per_step = F.kl_div(
                soft_student_seq, soft_teacher_expanded, reduction='none' 
            ).sum(dim=-1) # [B, T]
            
            weighted_kl = (kl_per_step * time_weights_truncated).sum(dim=-1)
            # 注意：使用标准化后，论文建议不再乘以 temperature^2，或直接设为 1
            seq_loss = weighted_kl.mean() 
            
            # 同样对最终输出 final_loss 应用标准化 
            final_loss = self._compute_kl_loss(
                self._logit_standardization(student_logits), 
                std_teacher_logits
            )
            # sigmoid_beta = torch.sigmoid(self.beta)
            # kl_loss = sigmoid_beta * seq_loss + (1 - sigmoid_beta) * final_loss
            kl_loss = self.beta * seq_loss + (1 - self.beta) * final_loss
            
            # ---- MTSKD 融合 ----
            # 将 MemKD 和 KL 损失加权组合
            
            # 方案选择：
            USE_LEARNABLE_WEIGHT = True  # 可以设置为True体验可学习权重
            
            if USE_LEARNABLE_WEIGHT:
                # 方案1：使用可学习权重
                memkd_weight = torch.sigmoid(self.mtskd_weight)
                kl_weight = 1.0 - torch.sigmoid(self.mtskd_weight)
                soft_loss = memkd_weight * memkd_loss + kl_weight * kl_loss
                # soft_loss = self.mtskd_weight * memkd_loss + (1 - self.mtskd_weight) * kl_loss
                
            else:
                # 方案2：使用固定权重（推荐，避免过多的可学习参数）
                soft_loss = 0.5 * memkd_loss + 0.5 * kl_loss    
        
        # 结合序列损失和最终输出损失
        # soft_loss = 0.5 * seq_loss + 0.5 * final_loss
        
        # 组合损失
        alpha = torch.sigmoid(self.alpha)
        # total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
        # total_loss = soft_loss + hard_loss
        # ========= 动态蒸馏权重 =========
        progress = self.current_epoch / self.total_epoch

        if progress < 0.2:
            kd_weight = 0.0          # 前期：只学CE
        elif progress < 0.6:
            kd_weight = (progress-0.2)/0.4   # 中期：逐渐增加
        else:
            kd_weight = 1.0          # 后期：强蒸馏

        ce_weight = 1.0 - 0.3 * kd_weight   # CE慢慢减弱

        total_loss = ce_weight * hard_loss + kd_weight * soft_loss
    
        
        return total_loss, hard_loss, soft_loss, alpha.item(), self.beta.item(), memkd_weight.item()
        # return total_loss, hard_loss, soft_loss, alpha.item(), self.beta.item(), self.mtskd_weight.item()