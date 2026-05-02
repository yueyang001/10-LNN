"""
测试对比蒸馏实现
"""

import sys
import torch
import os

# 添加路径
# 获取项目根目录（从 tests/ 目录回到 UATR/）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)  # 添加项目根目录
sys.path.append(os.path.join(project_root, 'experiments', '1_comparison'))

from distillation_loss import create_distillation_loss
from models.Audio_TeacherNet import build_Audio_TeacherNet
from models.LNN import AudioCfC
from models.distillation import AudioDistillationModel


def test_distillation_loss():
    """测试蒸馏损失函数"""
    print("="*60)
    print("Testing Distillation Loss Functions")
    print("="*60)

    # 测试基础KD
    kd_loss = create_distillation_loss('kd', temperature=4.0, alpha=0.5)
    print("✓ KD loss created successfully")

    # 测试LSKD
    lskd_loss = create_distillation_loss('lskd')
    print("✓ LSKD loss created successfully")

    # 测试DKD
    dkd_loss = create_distillation_loss('dkd', temperature=4.0, alpha=0.5,
                                       dkd_alpha=1.0, dkd_beta=1.0)
    print("✓ DKD loss created successfully")

    # 测试RKD
    rkd_loss = create_distillation_loss('rkd', rkd_wd=25, rkd_wa=50)
    print("✓ RKD loss created successfully")

    # 测试AT
    at_loss = create_distillation_loss('at', at_p=2)
    print("✓ AT loss created successfully")

    # 测试损失计算
    student_logits = torch.randn(4, 4)  # batch=4, num_classes=4
    teacher_logits = torch.randn(4, 4)
    labels = torch.randint(0, 4, (4,))

    # 测试KD
    loss = kd_loss(student_logits, teacher_logits, labels)
    print(f"✓ KD loss computed: {loss.item():.4f}")

    # 测试LSKD
    loss = lskd_loss(student_logits, teacher_logits, 4.0)
    print(f"✓ LSKD loss computed: {loss.item():.4f}")

    print("\n" + "="*60)
    print("All loss functions tested successfully!")
    print("="*60)


def test_model_forward():
    """测试模型前向传播"""
    print("\n" + "="*60)
    print("Testing Model Forward Pass")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建模型
    model = AudioDistillationModel(
        num_classes=4,
        teacher_pretrained=False,  # 测试时不需要预训练权重
        freeze_teacher=True,
        teacher_checkpoint=None,
        p_encoder=0.2,
        p_classifier=0.3
    ).to(device)

    print("✓ AudioDistillationModel created")

    # 创建输入
    audio_student = torch.randn(2, 1, 48000).to(device)
    audio_teacher = torch.randn(2, 1, 48000).to(device)

    print(f"✓ Input created: student={audio_student.shape}, teacher={audio_teacher.shape}")

    # 前向传播
    with torch.no_grad():
        student_logits, stu_seq_logits, fl, fg, bl, bg, x_encoder, \
            teacher_logits, output_cnn_features, teacher_all_hidden_states = \
            model([audio_student, audio_teacher])

    print(f"✓ Forward pass completed")
    print(f"  - student_logits: {student_logits.shape}")
    print(f"  - stu_seq_logits: {stu_seq_logits.shape}")
    print(f"  - x_encoder: {x_encoder.shape}")
    print(f"  - teacher_logits: {teacher_logits.shape}")
    if output_cnn_features is not None:
        print(f"  - output_cnn_features: {output_cnn_features.shape}")
    if teacher_all_hidden_states is not None:
        print(f"  - teacher_all_hidden_states: {teacher_all_hidden_states.shape}")

    print("\n" + "="*60)
    print("Model forward pass tested successfully!")
    print("="*60)


def test_integration():
    """测试完整流程"""
    print("\n" + "="*60)
    print("Testing Complete Pipeline")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建模型
    model = AudioDistillationModel(
        num_classes=4,
        teacher_pretrained=False,
        freeze_teacher=True,
        teacher_checkpoint=None,
        p_encoder=0.2,
        p_classifier=0.3
    ).to(device)

    # 创建损失函数
    criterion = create_distillation_loss(
        'kd',
        temperature=4.0,
        alpha=0.5
    ).to(device)

    print("✓ Model and criterion created")

    # 创建输入
    audio_student = torch.randn(2, 1, 48000).to(device)
    audio_teacher = torch.randn(2, 1, 48000).to(device)
    labels = torch.randint(0, 4, (2,)).to(device)

    print(f"✓ Inputs and labels created")

    # 前向传播
    student_logits, stu_seq_logits, fl, fg, bl, bg, x_encoder, \
        teacher_logits, output_cnn_features, teacher_all_hidden_states = \
        model([audio_student, audio_teacher])

    print("✓ Forward pass completed")

    # 计算损失
    loss = criterion(student_logits, teacher_logits, labels)

    print(f"✓ Loss computed: {loss.item():.4f}")

    # 反向传播
    loss.backward()

    print("✓ Backward pass completed")

    print("\n" + "="*60)
    print("Complete pipeline tested successfully!")
    print("="*60)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting Comparison Distillation Tests")
    print("="*60)

    try:
        test_distillation_loss()
        test_model_forward()
        test_integration()

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)

        print("\nYou can now run the training script:")
        print("  python run_single_comparison.py --method kd --gpus 0,1")
        print("\nOr run batch experiments:")
        print("  ./run_comparison_experiments.sh")

    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
