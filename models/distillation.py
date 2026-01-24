import torch
import torch.nn as nn
# from .CFC import AudioCfC
from .LNN import AudioCfC
from .Audio_TeacherNet import build_Audio_TeacherNet


class AudioDistillationModel(nn.Module):
    """音频知识蒸馏模型"""
    def __init__(
        self,
        num_classes=4,
        teacher_pretrained=True,
        freeze_teacher=True,
        teacher_checkpoint=None,
        # 学生网络参数
        cfc_input_size=32,
        cfc_output_size=32,
        cfc_hidden_size=64,
        pooling='last',
    ):
        super().__init__()
        
        # 教师网络 (ViT)
        self.teacher = build_Audio_TeacherNet(
            num_classes=num_classes, 
            checkpoint_path=teacher_checkpoint
        )
        
        # 冻结教师网络
        if freeze_teacher:
            for param in self.teacher.parameters():
                param.requires_grad = False
            self.teacher.eval()
        
        # 学生网络 (LNN AudioCfC)
        self.student = AudioCfC(num_classes=num_classes)
        # 确保学生网络输出维度匹配
        # self.student.head = nn.Linear(cfc_output_size, num_classes)
        # seq_len = 62
        # weights = torch.arange(1, seq_len + 1, dtype=torch.float32)  # [1, 2, 3, ..., 62]
        # weights = weights / weights.sum()  # 归一化，使权重和为1
        # self.distill_para = nn.Parameter(weights.view(1, seq_len, 1), requires_grad=True)
        self.num_classes = num_classes
        self.stu_linear = nn.Linear(cfc_hidden_size, num_classes)

        
        self.freeze_teacher = freeze_teacher
    
    def forward(self, audio_input):
        """
        Args:
            audio_input: 音频输入 [B, 1, 48000]
        Returns:
            student_logits: 学生网络输出
            teacher_logits: 教师网络输出 (推理时为 None)
        """
        # 学生网络前向传播
        student_logits, stu_sequence_logits = self.student(audio_input[0])
        stu_sequence_logits = self.stu_linear(stu_sequence_logits)  # [bs, 62, 4]
        # 计算加权的学生序列输出
        # stu_sequence_logits = stu_sequence_logits * self.distill_para  # [bs, 62, 4]
        # stu_sequence_logits = torch.sum(stu_sequence_logits, dim=1)  # [bs, 4]
        # TODO：应该是有两个方法，一个是直接用给sequence不同的权重再分别和教师计算KL，一个是直接用加权的序列输出和教师计算KL。
        # 就是一种是[bs, 62, 4]和[bs, 62, 4]计算KL，一种参与loss计算的是[bs, 62, 4]，一种是[bs, 4]。
        # 对应的distill loss应该也需要改一下
        
        # 教师网络前向传播 (仅训练时)
        teacher_logits = None


        with torch.no_grad():
            teacher_logits, _, _ = self.teacher(audio_input[1])

        
        return student_logits, stu_sequence_logits, teacher_logits


if __name__ == "__main__":
    # 测试代码
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = AudioDistillationModel(num_classes=4, 
                                   teacher_checkpoint='/media/hdd1/fubohan/Code/UATR/checkpoints/deepship_vit/best_model.pth')
    model.to(device)
    
    audio = torch.randn(2, 1, 48000).to(device)
    image = torch.randn(2, 3, 224, 224).to(device)
    
    student_out, stu_sequence_logits, teacher_out = model(audio, image)
    print(f"学生网络输出: {student_out.shape}")
    print(f"学生网络sequence输出: {stu_sequence_logits.shape}")
    print(f"教师网络输出: {teacher_out.shape}")