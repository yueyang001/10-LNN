"""
测试 MTSKD_Temp 蒸馏损失函数
"""
import torch
import sys
import os
import torch.nn as nn
# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.distillation_loss import DistillationLoss
from models.LNN import AudioCfC
from models.Audio_TeacherNet import build_Audio_TeacherNet


def test_mtskd_temp():
    """测试 MTSKD_Temp 蒸馏方法"""
    print("=" * 80)
    print("测试 MTSKD_Temp 蒸馏损失函数")
    print("=" * 80)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试参数设置
    batch_size = 4
    num_classes = 5
    seq_len = 16
    audio_length = 48000
    
    # 创建蒸馏损失函数
    criterion = DistillationLoss(
        temperature=2.0,
        alpha=0.3,
        learnable_alpha=True,
        weight_type='uniform',
        distill_type='MTSKD_Temp',
        seq_len=seq_len,
        num_classes=num_classes,
        use_dynamic=False
    ).to(device)
    
    print(f"\n蒸馏损失函数配置:")
    print(f"  - distill_type: MTSKD_Temp")
    print(f"  - temperature: 2.0")
    print(f"  - alpha: 0.3 (learnable)")
    print(f"  - num_classes: {num_classes}")
    print(f"  - seq_len: {seq_len}")
    
    # 创建学生网络
    student_model = AudioCfC(
        num_classes=num_classes,
        p_encoder=0.2,
        p_classifier=0.3
    ).to(device)
    
    # 添加线性层将时序特征转换为 logits (与 distillation.py 中的 self.stu_linear 相同)
    stu_linear = nn.Linear(64, num_classes).to(device)
    
    # 创建教师网络（使用 ShipsEar 的 checkpoint）
    teacher_model = build_Audio_TeacherNet(
        num_classes=num_classes,
        checkpoint_path='/media/hdd1/fubohan/Code/UATR/models/Audio_Teacher_ShipsEar_622/checkpoints/Student.pth'
    ).to(device)
    teacher_model.eval()
    
    print(f"\n学生网络参数量: {sum(p.numel() for p in student_model.parameters()):,}")
    print(f"教师网络参数量: {sum(p.numel() for p in teacher_model.parameters()):,}")
    
    # 生成测试数据
    audio_input = torch.randn(batch_size, 1, audio_length).to(device)
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    print(f"\n输入数据形状:")
    print(f"  - audio_input: {audio_input.shape}")
    print(f"  - labels: {labels.shape}")
    
    # 学生网络前向传播
    print(f"\n执行学生网络前向传播...")
    student_logits, stu_sequence_logits_raw, fl, fg, bl, bg, x_encoder = student_model(audio_input)
    
    # 将时序特征转换为 logits (与 distillation.py 中的处理相同)
    stu_sequence_logits = stu_linear(stu_sequence_logits_raw)  # [B, 16, 64] -> [B, 16, 5]
    
    print(f"学生网络输出形状:")
    print(f"  - student_logits: {student_logits.shape} [B, num_classes]")
    print(f"  - stu_sequence_logits_raw: {stu_sequence_logits_raw.shape} [B, seq_len, 64]")
    print(f"  - stu_sequence_logits: {stu_sequence_logits.shape} [B, seq_len, num_classes]")
    print(f"  - fl (fwd_local): {fl.shape} [B, seq_len, 64]")
    print(f"  - fg (fwd_global): {fg.shape} [B, seq_len, 64]")
    print(f"  - bl (bwd_local): {bl.shape} [B, seq_len, 64]")
    print(f"  - bg (bwd_global): {bg.shape} [B, seq_len, 64]")
    print(f"  - x_encoder: {x_encoder.shape} [B, seq_len, 64]")
    
    # 教师网络前向传播
    print(f"\n执行教师网络前向传播...")
    with torch.no_grad():
        teacher_logits, output_cnn_features, teacher_all_hidden_states, _ = teacher_model(audio_input)
    
    print(f"教师网络输出形状:")
    print(f"  - teacher_logits: {teacher_logits.shape} [B, num_classes]")
    print(f"  - output_cnn_features: {output_cnn_features.shape} [B, 149, 512]")
    print(f"  - teacher_all_hidden_states: {teacher_all_hidden_states.shape} [13, B, 149, 1024]")
    
    # 计算 MTSKD_Temp 损失
    print(f"\n计算 MTSKD_Temp 蒸馏损失...")
    loss, hard_loss, soft_loss, alpha, beta, memkd_weight = criterion(
        student_logits, stu_sequence_logits, fl, fg, bl, bg, x_encoder,
        teacher_logits, output_cnn_features, labels, teacher_all_hidden_states
    )
    
    print(f"\n损失结果:")
    print(f"  - total_loss: {loss.item():.4f}")
    print(f"  - hard_loss (CE): {hard_loss.item():.4f}")
    print(f"  - soft_loss (KD): {soft_loss.item():.4f}")
    print(f"  - alpha (sigmoid): {alpha:.4f}")
    print(f"  - beta: {beta:.4f}")
    print(f"  - memkd_weight: {memkd_weight:.4f}")
    
    # 验证损失是否为数值
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"\n  警告: 损失值异常 (NaN 或 Inf)!")
    else:
        print(f"\n✓ 损失计算成功，值正常")
    
    # 预测结果
    print(f"\n预测结果 (学生网络):")
    pred = student_logits.argmax(dim=1)
    correct = (pred == labels).sum().item()
    accuracy = 100. * correct / batch_size
    print(f"  - 预测标签: {pred.tolist()}")
    print(f"  - 真实标签: {labels.tolist()}")
    print(f"  - 正确数: {correct}/{batch_size}")
    print(f"  - 准确率: {accuracy:.2f}%")
    
    # 测试梯度反向传播
    print(f"\n测试梯度反向传播...")
    loss.backward()
    print(f"✓ 梯度计算成功")
    
    # 检查梯度
    has_grad = 0
    total_grad = 0
    for name, param in student_model.named_parameters():
        if param.grad is not None:
            has_grad += 1
            total_grad += 1
        else:
            total_grad += 1
    
    print(f"  - 有梯度的参数: {has_grad}/{total_grad}")
    
    # 测试可学习参数
    print(f"\n可学习参数:")
    print(f"  - criterion.alpha: {criterion.alpha.item():.4f} (requires_grad: {criterion.alpha.requires_grad})")
    print(f"  - criterion.beta: {criterion.beta.item():.4f} (requires_grad: {criterion.beta.requires_grad})")
    print(f"  - criterion.mtskd_weight: {criterion.mtskd_weight.item():.4f} (requires_grad: {criterion.mtskd_weight.requires_grad})")
    
    print(f"\n{'=' * 80}")
    print(f"MTSKD_Temp 测试完成!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    test_mtskd_temp()
