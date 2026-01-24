import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """知识蒸馏损失函数 - 支持 KL 和 MSE 两种模式"""
    def __init__(self, temperature=4.0, alpha=0.5, learnable_alpha=True, 
                 seq_len=250, weight_type='uniform', distill_type='base'):
        """
        Args:
            distill_type: 'kl' 或 'mse'
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
        
        weights = self._create_weights(seq_len, weight_type)
        self.register_buffer('time_weights', weights)
    
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
    
    def forward(self, student_logits, stu_sequence_logits, teacher_logits, labels):
        B, seq_len, num_classes = stu_sequence_logits.shape
        
        # 硬标签损失
        hard_loss = self.ce_loss(student_logits, labels)
        
        if self.distill_type == 'kl':
            # ============ KL 散度方法 ================
            soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
            soft_teacher_expanded = soft_teacher.unsqueeze(1).expand(-1, seq_len, -1)
            soft_student_seq = F.log_softmax(stu_sequence_logits / self.temperature, dim=-1)
            
            kl_per_step = F.kl_div(
                soft_student_seq, soft_teacher_expanded, reduction='none'
            ).sum(dim=-1)
            
            weighted_kl = (kl_per_step * self.time_weights).sum(dim=-1)
            seq_loss = weighted_kl.mean() * (self.temperature ** 2)
            
            # 最终输出的 KL
            final_loss = self._compute_kl_loss(student_logits, teacher_logits)
            soft_loss = self.beta * seq_loss + (1 - self.beta) * final_loss
            
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
        elif self.distill_type == 'base':
            soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
            soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
            soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
            soft_loss = soft_loss * (self.temperature ** 2)
      
        
        else:
            raise ValueError(f"Unknown distill_type: {self.distill_type}")
        
        # 结合序列损失和最终输出损失
        # soft_loss = 0.5 * seq_loss + 0.5 * final_loss
        
        # 组合损失
        alpha = torch.sigmoid(self.alpha)
        total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
        
        return total_loss, hard_loss, soft_loss, alpha.item(), self.beta.item()