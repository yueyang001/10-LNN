from sympy import beta
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class DistillationLoss(nn.Module):
    """知识蒸馏损失函数 - 支持 KL 和 MSE 两种模式"""
    def __init__(self, temperature=4.0, alpha=0.5, learnable_alpha=True, 
                 seq_len=16, weight_type='uniform', distill_type='base', use_dynamic=False, num_classes=4):
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
        self.mtskd_weight = nn.Parameter(torch.tensor(0.0))

        weights = self._create_weights(seq_len, weight_type) # weight: torch.Size([16])
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
            nn.Linear(256, num_classes)
        )
        self.current_epoch = 0
        self.total_epoch = 100  # 默认，可从外部传
        self.use_dynamic = use_dynamic

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
        def logit_std(x):
            return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-5)

        std_student = logit_std(student_logits)
        std_teacher = logit_std(teacher_logits)

        # 2. 计算 Softmax
        soft_teacher = F.softmax(std_teacher, dim=-1)
        soft_student = F.log_softmax(std_student, dim=-1)

        # 3. 计算 KL 散度
        kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        return kl_loss
    
    def _compute_mse_loss(self, student_logits, teacher_logits):
        """计算 MSE 损失"""
        return self.mse_loss(student_logits, teacher_logits)
    
    def _compute_memkd_loss(self, student_h, teacher_h, z):
        """
        student_h: [B, 16, 64]
        teacher_h: [B, 149, 512]
        z: 偏移量
        """
        # ---- 步骤 A: 时间步对齐 (教师 149 -> 16) ----
        t_h = teacher_h.permute(0, 2, 1) # [B, 512, 149]
        t_h = F.interpolate(t_h, size=student_h.size(1), mode='linear', align_corners=True)
        t_h = t_h.permute(0, 2, 1) # [B, 16, 512]
        
        s_h = student_h # [B, 16, 64]
        
        # ---- 步骤 B: 计算内存差异 ----
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
        
        # ---- 步骤 C: 归一化 ----
        eps = 1e-8
        norm_t = torch.norm(h_t_teacher, p=2, dim=-1, keepdim=True) + eps
        f_t = delta_teacher / norm_t
        
        norm_s = torch.norm(h_t_student, p=2, dim=-1, keepdim=True) + eps
        f_s = delta_student / norm_s
        
        # ---- 步骤 D: 计算幅度差异 ----
        mag_t = torch.norm(f_t, p=2, dim=-1)
        mag_s = torch.norm(f_s, p=2, dim=-1)

        # 使用 Smooth L1 Loss
        loss = F.smooth_l1_loss(mag_s, mag_t)
        return loss
    
    def _logit_standardization(self, logits, dim=-1):
        """Logit Standardization (Z-score) 预处理"""
        mean = logits.mean(dim=dim, keepdim=True)
        std = logits.std(dim=dim, keepdim=True) + 1e-5
        return (logits - mean) / std

    def forward(self, student_logits, stu_seq_logits, fl, fg, bl, bg, x_encoder, teacher_logits, output_cnn_features, labels, teacher_all_hidden_states=None):
        # ========== 核心修复1：初始化所有关键变量，避免未定义 ==========
        device = student_logits.device
        memkd_weight = torch.tensor(0.0, device=device, dtype=torch.float32)
        soft_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        seq_len = stu_seq_logits.size(1)  # 从输入获取真实序列长度，避免硬编码
        
        # 硬标签损失
        hard_loss = self.ce_loss(student_logits, labels)
        
        # ========== 不同蒸馏类型分支处理 ==========
        if self.distill_type == 'kl':
            """时序 KL 散度方法 (Logit Standardization)"""
            # 初始化memkd_weight
            memkd_weight = torch.tensor(0.0, device=device)
            
            # 应用 Logit Standardization
            std_teacher_logits = self._logit_standardization(teacher_logits)
            std_stu_seq_logits = self._logit_standardization(stu_seq_logits)

            # 计算 Softmax
            soft_teacher = F.softmax(std_teacher_logits, dim=-1)
            soft_teacher_expanded = soft_teacher.unsqueeze(1).expand(-1, seq_len, -1)
            soft_student_seq = F.log_softmax(std_stu_seq_logits, dim=-1)

            # 时间权重处理
            time_weights_truncated = self.time_weights[:seq_len].to(device)

            # 计算 KL 散度
            kl_per_step = F.kl_div(
                soft_student_seq, soft_teacher_expanded, reduction='none' 
            ).sum(dim=-1)
            weighted_kl = (kl_per_step * time_weights_truncated).sum(dim=-1)
            seq_loss = weighted_kl.mean()
            
            # 最终输出的 KL 损失
            final_loss = self._compute_kl_loss(
                self._logit_standardization(student_logits), 
                std_teacher_logits
            )
            
            soft_loss = self.beta * seq_loss + (1 - self.beta) * final_loss
        
        elif self.distill_type == 'Tser':
            """特征对齐与时序 KL 蒸馏"""
            # 初始化memkd_weight
            memkd_weight = torch.tensor(0.0, device=device)
            
            # 检查教师特征是否存在
            if teacher_all_hidden_states is None:
                raise ValueError("Tser模式需要teacher_all_hidden_states")
            
            # 1. 线性投影：教师 1024 维 -> num_classes 维
            projected_teacher = self.feat_projector(teacher_all_hidden_states)

            # 2. 时间维度对齐：149 -> 16
            projected_teacher = projected_teacher.transpose(1, 2) 
            aligned_teacher_feat = F.interpolate(
                projected_teacher, 
                size=seq_len, 
                mode='linear', 
                align_corners=False
            )
            aligned_teacher_feat = aligned_teacher_feat.transpose(1, 2)

            # 3. 计算 KL 散度
            soft_teacher = F.softmax(aligned_teacher_feat / self.temperature, dim=-1)
            soft_student_seq = F.log_softmax(stu_seq_logits / self.temperature, dim=-1)

            kl_per_step = F.kl_div(
                soft_student_seq, 
                soft_teacher, 
                reduction='none'
            ).sum(dim=-1)
            
            # 时间权重处理
            time_weights_truncated = self.time_weights[:seq_len].to(device)
            weighted_kl = (kl_per_step * time_weights_truncated).sum(dim=-1)
            seq_loss = weighted_kl.mean() * (self.temperature ** 2)

            # 最终输出的 KL 损失
            final_loss = self._compute_kl_loss(student_logits, teacher_logits)
            soft_loss = self.beta * seq_loss + (1 - self.beta) * final_loss
            
        elif self.distill_type == 'MemKD':
            """MemKD 核心实现"""
            # 检查输入特征
            if x_encoder is None or output_cnn_features is None:
                raise ValueError("MemKD需要x_encoder和output_cnn_features")
            
            # 1. 计算短程内存差异损失 (z=1)
            loss_short = self._compute_memkd_loss(x_encoder, output_cnn_features, z=1)
            
            # 2. 计算远程内存差异损失 (z 随机)
            T_max = x_encoder.size(1)
            z_random = random.randint(2, T_max // 2)
            loss_long = self._compute_memkd_loss(x_encoder, output_cnn_features, z=z_random)
            
            # 3. 组合损失
            soft_loss = 0.5 * loss_short + 1.0 * loss_long
            # 赋值memkd_weight（默认0，也可自定义）
            memkd_weight = torch.tensor(0.0, device=device)

        elif self.distill_type == 'mse':
            """MSE 方法"""
            # 初始化memkd_weight
            memkd_weight = torch.tensor(0.0, device=device)
            
            teacher_expanded = teacher_logits.unsqueeze(1).expand(-1, seq_len, -1)
            
            # 每个时间步的 MSE
            mse_per_step = ((stu_seq_logits - teacher_expanded) ** 2).mean(dim=-1)
            
            # 加权求和
            time_weights_truncated = self.time_weights[:seq_len].to(device)
            weighted_mse = (mse_per_step * time_weights_truncated).sum(dim=-1)
            seq_loss = weighted_mse.mean()
            
            # 最终输出的 MSE
            final_loss = self._compute_mse_loss(student_logits, teacher_logits)
            soft_loss = self.beta * seq_loss + (1 - self.beta) * final_loss
            
        elif self.distill_type == 'base':
            """基础 KL 蒸馏"""
            memkd_weight = torch.tensor(0.0, device=device)
            soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
            soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
            soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
            soft_loss = soft_loss * (self.temperature ** 2)
            
        elif self.distill_type == 'MTSKD':
            """MTSKD (MemKD + KL) 混合蒸馏"""
            # 检查输入特征
            if x_encoder is None or output_cnn_features is None:
                raise ValueError("MTSKD需要x_encoder和output_cnn_features")
            
            # ---- MemKD 部分 ----
            memkd_loss_short = self._compute_memkd_loss(x_encoder, output_cnn_features, z=1)
            T_max = x_encoder.size(1)
            z_random = random.randint(2, T_max // 2)
            memkd_loss_long = self._compute_memkd_loss(x_encoder, output_cnn_features, z=z_random)
            memkd_loss = 0.5 * memkd_loss_short + 1.0 * memkd_loss_long
            
            # ---- KL 部分 ----
            soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
            soft_teacher_expanded = soft_teacher.unsqueeze(1).expand(-1, seq_len, -1)
            soft_student_seq = F.log_softmax(stu_seq_logits / self.temperature, dim=-1)
            
            time_weights_truncated = self.time_weights[:seq_len].to(device)
            kl_per_step = F.kl_div(
                soft_student_seq, soft_teacher_expanded, reduction='none'
            ).sum(dim=-1)
            weighted_kl = (kl_per_step * time_weights_truncated).sum(dim=-1)
            seq_loss = weighted_kl.mean() * (self.temperature ** 2)
            
            final_loss = self._compute_kl_loss(student_logits, teacher_logits)
            kl_loss = self.beta * seq_loss + (1 - self.beta) * final_loss
            
            # ---- MTSKD 融合 ----
            USE_LEARNABLE_WEIGHT = True
            if USE_LEARNABLE_WEIGHT:
                memkd_weight = torch.sigmoid(self.mtskd_weight)  # 关键：赋值memkd_weight
                kl_weight = 1.0 - memkd_weight
                soft_loss = memkd_weight * memkd_loss + kl_weight * kl_loss
            else:
                soft_loss = 0.5 * memkd_loss + 0.5 * kl_loss
                memkd_weight = torch.tensor(0.5, device=device)  # 固定权重
                
        elif self.distill_type == 'MTSKD_Temp':
            """MTSKD_Temp (MemKD + KL_Temp) 混合蒸馏"""
            # 检查输入特征
            if x_encoder is None or output_cnn_features is None:
                raise ValueError("MTSKD_Temp需要x_encoder和output_cnn_features")
            
            # ---- MemKD 部分 ----
            memkd_loss_short = self._compute_memkd_loss(x_encoder, output_cnn_features, z=1)
            T_max = x_encoder.size(1)
            z_random = random.randint(2, T_max // 2)
            memkd_loss_long = self._compute_memkd_loss(x_encoder, output_cnn_features, z=z_random)
            memkd_loss = 0.5 * memkd_loss_short + 1.0 * memkd_loss_long
            
            # ---- KL_Temp 部分 (Logit Standardization) ----
            std_teacher_logits = self._logit_standardization(teacher_logits)
            std_stu_seq_logits = self._logit_standardization(stu_seq_logits)

            soft_teacher = F.softmax(std_teacher_logits, dim=-1)
            soft_teacher_expanded = soft_teacher.unsqueeze(1).expand(-1, seq_len, -1)
            soft_student_seq = F.log_softmax(std_stu_seq_logits, dim=-1)

            time_weights_truncated = self.time_weights[:seq_len].to(device)
            kl_per_step = F.kl_div(
                soft_student_seq, soft_teacher_expanded, reduction='none' 
            ).sum(dim=-1)
            weighted_kl = (kl_per_step * time_weights_truncated).sum(dim=-1)
            seq_loss = weighted_kl.mean() 
            
            final_loss = self._compute_kl_loss(
                self._logit_standardization(student_logits), 
                std_teacher_logits
            )
            kl_loss = self.beta * seq_loss + (1 - self.beta) * final_loss
            
            # ---- MTSKD 融合 ----
            USE_LEARNABLE_WEIGHT = True
            if USE_LEARNABLE_WEIGHT:
                memkd_weight = torch.sigmoid(self.mtskd_weight)  # 关键：赋值memkd_weight
                kl_weight = 1.0 - memkd_weight
                soft_loss = memkd_weight * memkd_loss + kl_weight * kl_loss
            else:
                soft_loss = 0.5 * memkd_loss + 0.5 * kl_loss
                memkd_weight = torch.tensor(0.5, device=device)
        
        # ========== 动态蒸馏权重 ==========
        progress = self.current_epoch / self.total_epoch
        if self.use_dynamic:
            if progress < 0.2:
                kd_weight = 0.0          
            elif progress < 0.6:
                kd_weight = (progress-0.2)/0.4   
            else:
                kd_weight = 1.0          
        else:
            kd_weight = 1.0

        # ========== 组合最终损失 ==========
        alpha = torch.sigmoid(self.alpha)
        total_loss = hard_loss + kd_weight * soft_loss
    
        # ========== 返回所有变量（确保都已定义） ==========
        return total_loss, hard_loss, soft_loss, alpha.item(), self.beta.item(), memkd_weight.item()