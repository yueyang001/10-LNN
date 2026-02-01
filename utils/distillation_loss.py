import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import logging
import os
# 在类/模块最上方（或 __init__ 中）初始化 logger
logger = logging.getLogger("MemKD_magnitude")
logger.setLevel(logging.INFO)

# 只在第一次运行时添加 handler（避免重复输出）
if not logger.handlers:
    # 可以输出到文件 + 控制台
    log_dir = "logs/memkd"
    os.makedirs(log_dir, exist_ok=True)
    
    file_handler = logging.FileHandler(os.path.join(log_dir, "magnitude_track.log"), mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
class DistillationLoss(nn.Module):
    """知识蒸馏损失函数 - 支持 KL 和 MSE 两种模式"""
    def __init__(self, temperature=4.0, alpha=0.5, learnable_alpha=True, 
                 seq_len=16, weight_type='uniform', distill_type='base', num_classes=4):
        """
        Args:
            distill_type: 'kl' 或 'mse'或'MemKD'
        """
        super().__init__()
        self.temperature = temperature
        self.distill_type = distill_type
        
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer('alpha', torch.tensor(alpha))
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

        self.beta = nn.Parameter(torch.tensor(0.5))
        
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
        """计算 KL 散度损失"""
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        return kl_loss * (self.temperature ** 2)
    
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
        # # ─────────────── 记录幅度 ───────────────
        # # 取 batch 的 mean，便于观察趋势
        # mean_mag_t = mag_t.mean().item()
        # mean_mag_s = mag_s.mean().item()
        
        # logger.info(f"z={z:2d} | mag_t_mean={mean_mag_t:.6f} | mag_s_mean={mean_mag_s:.6f} | diff={mean_mag_s - mean_mag_t:+.6f}")
        # ────────────────────────────────────────
        # 使用 Smooth L1 Loss (论文推荐)
        loss = F.smooth_l1_loss(mag_s, mag_t)
        
        return loss
    
    def forward(self, student_logits, stu_sequence_logits, fl, fg, bl, bg, x_encoder, teacher_logits, output_cnn_features,labels, teacher_all_hidden_states=None):
        _, seq_len, _ = stu_sequence_logits.shape
    
        # 硬标签损失
        hard_loss = self.ce_loss(student_logits, labels)
        
        if self.distill_type == 'kl':
            # ============ 时序 KL 散度方法 ================
            """
            stu_sequence_logits:[batch,seq_len,num_classes] [B, 16, 5]
            teacher_logits: [batch, num_classes] 
            """
            soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1) # -> [B, C]
            soft_teacher_expanded = soft_teacher.unsqueeze(1).expand(-1, seq_len, -1) # -> [B, T, C] (通过广播机制扩展，每一帧都对齐同一个教师标签)
            soft_student_seq = F.log_softmax(stu_sequence_logits / self.temperature, dim=-1) # ->[B, T, C]
            
            kl_per_step = F.kl_div(
                soft_student_seq, soft_teacher_expanded, reduction='none' # [B, T, C]
            ).sum(dim=-1) # [B, T]
            
            # 确保权重与实际序列长度匹配
            batch_size, actual_seq_len = kl_per_step.shape
            # print(f"  权重截取前 kl_per_step.shape: {kl_per_step.shape}") # ->[4, 16]
            # time_weights_truncated = self.time_weights[:actual_seq_len]
            time_weights_truncated = self.time_weights
            # print(f"  截取后 time_weights_truncated.shape: {time_weights_truncated.shape}") # ->[16]

            weighted_kl = (kl_per_step * time_weights_truncated).sum(dim=-1)
            seq_loss = weighted_kl.mean() * (self.temperature ** 2)
            
            # 最终输出的 KL
            final_loss = self._compute_kl_loss(student_logits, teacher_logits)
            soft_loss = self.beta * seq_loss + (1 - self.beta) * final_loss
        # if self.distill_type == 'kl':
        #     # ============ KL 散度方法 20260126修改 ================
        #     soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1) # teacher_logits: [B, num_classes]
        #     soft_teacher_expanded = soft_teacher.unsqueeze(1).expand(-1, seq_len, -1)
        #     # soft_student_seq = F.log_softmax(stu_sequence_logits / self.temperature, dim=-1)
        #     # 临时创建一个投影层（64 -> num_classes），只在蒸馏时使用，挪到__init__中
        #     # proj = nn.Linear(64, teacher_logits.size(-1))  # 64 是学生的特征维度，teacher_logits.size(-1) 是 num_classes
        #     # proj = proj.to(fl.device)  # 移动到正确设备

        #     # 对每个流进行投影
        #     # fl = proj(fl)   # [B, seq_len, 64] -> [B, seq_len, num_classes]
        #     # fg = proj(fg)
        #     # bl = proj(bl)
        #     # bg = proj(bg)  
        #     fl = self.seq_proj(fl)
        #     fg = self.seq_proj(fg)
        #     bl = self.seq_proj(bl)
        #     bg = self.seq_proj(bg)

        #     # 计算log_softmax
        #     soft_student_seq = F.log_softmax(fl / self.temperature, dim=-1)
        #     soft_student_seq_fg = F.log_softmax(fg / self.temperature, dim=-1)
        #     soft_student_seq_bl = F.log_softmax(bl / self.temperature, dim=-1)
        #     soft_student_seq_bg = F.log_softmax(bg / self.temperature, dim=-1)

        #     # 计算kl散度
        #     kl_per_step = F.kl_div(
        #         soft_student_seq, soft_teacher_expanded, reduction='none'
        #     ).sum(dim=-1)
        #     kl_per_step_fg = F.kl_div(
        #         soft_student_seq_fg, soft_teacher_expanded, reduction='none'
        #     ).sum(dim=-1)
        #     kl_per_step_bl = F.kl_div(
        #         soft_student_seq_bl, soft_teacher_expanded, reduction='none'
        #     ).sum(dim=-1)
        #     kl_per_step_bg = F.kl_div(
        #         soft_student_seq_bg, soft_teacher_expanded, reduction='none'
        #     ).sum(dim=-1)

        #     # 时间加权 - 确保权重与实际序列长度匹配
        #     batch_size, actual_seq_len = kl_per_step.shape
        #     time_weights_truncated = self.time_weights[:actual_seq_len]
            
        #     weighted_kl = (kl_per_step * time_weights_truncated).sum(dim=-1)
        #     weighted_kl_fg = (kl_per_step_fg * time_weights_truncated).sum(dim=-1)
        #     weighted_kl_bl = (kl_per_step_bl * time_weights_truncated).sum(dim=-1)
        #     weighted_kl_bg = (kl_per_step_bg * time_weights_truncated).sum(dim=-1)

        #     # 序列损失
        #     seq_loss = (0.25 *weighted_kl.mean() \
        #                 + 0.25 *weighted_kl_fg.mean() \
        #                     + 0.25 *weighted_kl_bl.mean() \
        #                         + 0.25 *weighted_kl_bg.mean() )* (self.temperature ** 2)
            
        #     # 最终输出的 KL
        #     final_loss = self._compute_kl_loss(student_logits, teacher_logits)
        #     soft_loss = self.beta * seq_loss + (1 - self.beta) * final_loss
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
      
        
        
        # 结合序列损失和最终输出损失
        # soft_loss = 0.5 * seq_loss + 0.5 * final_loss
        
        # 组合损失
        alpha = torch.sigmoid(self.alpha)
        total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
        
        return total_loss, hard_loss, soft_loss, alpha.item(), self.beta.item()


def test_kl_distillation_shapes():
    """详细测试 KL 蒸馏中所有张量的形状"""
    print("详细 KL 蒸馏形状测试")
    print("=" * 60)
    
    # 参数设置
    batch_size = 4
    init_seq_len = 16  # 初始化时指定长度
    actual_seq_len = 16  # 实际输入长度
    num_classes = 5
    temperature = 2.0
    
    # 输入数据
    student_logits = torch.randn(batch_size, num_classes)
    stu_sequence_logits = torch.randn(batch_size, actual_seq_len, num_classes)
    teacher_logits = torch.randn(batch_size, num_classes)
    fl = torch.randn(batch_size, actual_seq_len, 64)
    fg = torch.randn(batch_size, actual_seq_len, 64)
    bl = torch.randn(batch_size, actual_seq_len, 64)
    bg = torch.randn(batch_size, actual_seq_len, 64)
    x_encoder = torch.randn(batch_size, actual_seq_len, 64)
    output_cnn_features = torch.randn(batch_size, 149, 512)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    print(f"输入张量形状:")
    print(f"  student_logits: {student_logits.shape}")
    print(f"  stu_sequence_logits: {stu_sequence_logits.shape}")
    print(f"  teacher_logits: {teacher_logits.shape}")
    print(f"  x_encoder: {x_encoder.shape}")
    print(f"  output_cnn_features: {output_cnn_features.shape}")
    print()
    
    # 创建 KL 蒸馏损失
    criterion = DistillationLoss(
        seq_len=init_seq_len,  # 16
        distill_type='kl',
        num_classes=num_classes,
        temperature=temperature
    )
    
    print(f"初始化权重:")
    print(f"  criterion.time_weights.shape: {criterion.time_weights.shape}")
    print()
    
    # 模拟 forward 过程中的关键步骤
    print("KL 蒸馏前向传播步骤:")
    print("-" * 40)
    
    # 获取实际序列长度
    _, seq_len, _ = stu_sequence_logits.shape
    print(f"1. 实际序列长度 seq_len: {seq_len}")
    
    # 教师 softmax
    soft_teacher = torch.softmax(teacher_logits / temperature, dim=-1)
    print(f"2. soft_teacher.shape: {soft_teacher.shape}")
    
    # 扩展教师输出
    soft_teacher_expanded = soft_teacher.unsqueeze(1).expand(-1, seq_len, -1)
    print(f"3. soft_teacher_expanded.shape: {soft_teacher_expanded.shape}")
    
    # 学生序列 softmax
    soft_student_seq = F.log_softmax(stu_sequence_logits / temperature, dim=-1)
    print(f"4. soft_student_seq.shape: {soft_student_seq.shape}")
    
    # KL 散度计算
    kl_per_step = F.kl_div(
        soft_student_seq, soft_teacher_expanded, reduction='none'
    ).sum(dim=-1)
    print(f"5. kl_per_step.shape: {kl_per_step.shape}")
    
    # 时间权重处理（重点：这是修复的关键）
    batch_size, actual_seq_len = kl_per_step.shape
    print(f"6. 权重截取前 kl_per_step.shape: {kl_per_step.shape}")
    print(f"   原始 time_weights.shape: {criterion.time_weights.shape}")
    
    time_weights_truncated = criterion.time_weights[:actual_seq_len]
    print(f"   截取后 time_weights_truncated.shape: {time_weights_truncated.shape}")
    
    # 加权 KL 计算
    weighted_kl = (kl_per_step * time_weights_truncated).sum(dim=-1)
    print(f"7. weighted_kl.shape: {weighted_kl.shape}")
    
    # 序列损失
    seq_loss = weighted_kl.mean() * (temperature ** 2)
    print(f"8. seq_loss: {seq_loss} (标量)")
    
    # 最终输出的 KL
    final_loss = criterion._compute_kl_loss(student_logits, teacher_logits)
    print(f"9. final_loss: {final_loss} (标量)")
    
    # 组合损失
    soft_loss = criterion.beta * seq_loss + (1 - criterion.beta) * final_loss
    print(f"10. soft_loss: {soft_loss}")
    
    print("\n✓ 所有形状测试通过！")
    print("-" * 60)
    
    # 完整的前向传播测试
    print("\n完整前向传播测试:")
    try:
        total_loss, hard_loss, soft_loss, alpha, beta = criterion(
            student_logits, stu_sequence_logits, 
            fl, fg, bl, bg, 
            x_encoder, teacher_logits, 
            output_cnn_features, labels
        )
        print(f"✓ 完整测试成功!")
        print(f"  total_loss: {total_loss.item():.4f}")
        print(f"  hard_loss: {hard_loss.item():.4f}")
        print(f"  soft_loss: {soft_loss.item():.4f}")
        print(f"  alpha: {alpha:.4f}")
        print(f"  beta: {beta:.4f}")
    except Exception as e:
        print(f"✗ 完整测试失败: {e}")
    
    return True


if __name__ == "__main__":
    """测试 KL 蒸馏方法并打印所有张量形状"""
    print("Testing KL Distillation with Shape Debug")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 测试参数 - 使用不同的初始化序列长度和实际序列长度
    batch_size = 4
    init_seq_len = 250  # 初始化时的序列长度
    actual_seq_len = 16  # 实际输入的序列长度
    num_classes = 5
    
    print(f"初始化序列长度: {init_seq_len}")
    print(f"实际序列长度: {actual_seq_len}")
    print()
    
    try:
        # 运行详细形状测试
        test_kl_distillation_shapes()
        print("\n所有测试完成！")
        print("=" * 60)
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()


