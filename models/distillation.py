import torch
import torch.nn as nn
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.LNN import AudioCfC
from models.Audio_TeacherNet import build_Audio_TeacherNet

class AudioDistillationModel(nn.Module):
    """音频知识蒸馏模型"""
    def __init__(
        self,
        num_classes=4,
        teacher_pretrained=True,
        freeze_teacher=True,
        teacher_checkpoint=None,
        # 学生网络参数
        p_encoder=0.2,
        p_classifier=0.3,
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
        self.student = AudioCfC(
            num_classes=num_classes,
            p_encoder=p_encoder,
            p_classifier=p_classifier
            )
        # 确保学生网络输出维度匹配
        # self.student.head = nn.Linear(cfc_output_size, num_classes)
        # seq_len = 62
        # weights = torch.arange(1, seq_len + 1, dtype=torch.float32)  # [1, 2, 3, ..., 62]
        # weights = weights / weights.sum()  # 归一化，使权重和为1
        # self.distill_para = nn.Parameter(weights.view(1, seq_len, 1), requires_grad=True)
        self.num_classes = num_classes
        self.stu_linear = nn.Linear(cfc_hidden_size, num_classes) # 线性转换，64->num_classes

        
        self.freeze_teacher = freeze_teacher
    
    def forward(self, audio_input):
        """
        Args:
            audio_input: 音频输入 [B, 1, 48000]
        Returns:
            student_logits: 学生网络输出
            stu_sequence_logits: 学生网络序列输出
            x_encoder: 编码器输出
            teacher_logits: 教师网络输出 (推理时为 None)
        """
        # 学生网络前向传播
        student_logits, stu_sequence_logits, fl, fg, bl, bg, x_encoder = self.student(audio_input[0])
        # student_all_hidden_states = stu_sequence_logits
        stu_sequence_logits = self.stu_linear(stu_sequence_logits)  # [bs, 62, 4]
        # 计算加权的学生序列输出
        # stu_sequence_logits = stu_sequence_logits * self.distill_para  # [bs, 62, 4]
        # stu_sequence_logits = torch.sum(stu_sequence_logits, dim=1)  # [bs, 4]
        # TODO：应该是有两个方法，一个是直接用给sequence不同的权重再分别和教师计算KL，一个是直接用加权的序列输出和教师计算KL。
        # 就是一种是[bs, 62, 4]和[bs, 62, 4]计算KL，一种参与loss计算的是[bs, 62, 4]，一种是[bs, 4]。
        # 对应的distill loss应该也需要改一下
        
        # 教师网络前向传播 (仅训练时)
        teacher_logits = None
        teacher_all_hidden_states = None

        with torch.no_grad():
            teacher_logits, output_cnn_features, teacher_all_hidden_states, _ = self.teacher(audio_input[1])

        
        return student_logits, stu_sequence_logits, fl, fg, bl, bg, x_encoder, teacher_logits, output_cnn_features, teacher_all_hidden_states
        # return student_logits, stu_sequence_logits, fl, fg, bl, bg, x_encoder, teacher_logits, output_cnn_features, teacher_all_hidden_states, student_all_hidden_states

if __name__ == "__main__":
    # 测试代码
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = AudioDistillationModel(num_classes=5, 
                                #    teacher_checkpoint='/media/hdd1/fubohan/Code/UATR/models/Audio_Teacher_DeepShip_622/checkpoints/Student.pth')
                                   teacher_checkpoint='/media/hdd1/fubohan/Code/UATR/models/Audio_Teacher_ShipsEar_622/checkpoints/Student.pth')
    model.to(device)
    
    audio_student = torch.randn(2, 1, 48000).to(device)
    audio_teacher = torch.randn(2, 1, 48000).to(device)
    
    student_logits, stu_sequence_logits, _, _, _, _, _, teacher_logits, teacher_output_cnn_features, teacher_all_hidden_states = model([audio_student, audio_teacher])
    print(f"学生网络输出: {student_logits.shape}") # [2, 5] [batch,num_classes]
    print(f"学生网络结果:{student_logits}")
    print(f"学生网络sequence_logits输出: {stu_sequence_logits.shape}") # [2, 16, 5] [batch,seq_len,num_classes]
    print(f"教师网络输出: {teacher_logits.shape}")# [2, 5] [batch,num_classes]
    if teacher_output_cnn_features is not None:
        print(f"教师网络cnn编码后隐藏状态: {teacher_output_cnn_features.shape}")
    if teacher_all_hidden_states is not None:
        print(f"教师网络隐藏状态: {teacher_all_hidden_states.shape}")