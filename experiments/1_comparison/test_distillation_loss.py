#!/usr/bin/env python3
"""
测试蒸馏损失函数库 - 仅包含指定的20种方法
Logit-based (6种): KD, DKD, NKD, WSLD, LSKD, FreeKD
Feature-based (14种): AT, FSP, SP, PKT, RKD, NST, VID, CC, ICKD, MGD, SDD, VkD, DiffKD, CAT_KD, MKD
"""
import warnings
warnings.filterwarnings('ignore', message='pkg_resources is deprecated as an API')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from distillation_loss import (
    KDLoss, LSKDLoss, DKDLoss, NKDLoss, WSLDLoss,
    AttentionLoss, NSTLoss, FSPLoss, PKTLoss, RKDLoss,
    SimilarityLoss, CorrelationLoss, MGDLoss,
    CAT_KDLoss, ICKDLoss, VKDLoss, VIDLoss, create_distillation_loss
)
import torch
import torch.nn.functional as F


def test_logit_losses():
    """测试logit损失函数 - 仅包含6种指定方法"""
    print("=" * 60)
    print("测试Logit-based蒸馏方法 (6种)")
    print("=" * 60)

    # 准备测试数据
    batch_size = 4
    num_classes = 4
    student_logits = torch.randn(batch_size, num_classes)
    teacher_logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))

    # KD
    kd = KDLoss(temperature=4.0, alpha=0.5)
    loss_kd = kd(student_logits, teacher_logits, labels)
    print(f"✓ KD (基础蒸馏): {loss_kd.item():.4f}")

    # DKD
    dkd = DKDLoss(alpha=1.0, beta=1.0, temperature=4.0)
    loss_dkd = dkd(student_logits, teacher_logits, labels)
    print(f"✓ DKD (解耦蒸馏): {loss_dkd.item():.4f}")

    # NKD
    nkd = NKDLoss(temp=1.0, gamma=1.5)
    loss_nkd = nkd(student_logits, teacher_logits, labels)
    print(f"✓ NKD (负知识蒸馏): {loss_nkd.item():.4f}")

    # WSLD
    wsld = WSLDLoss(temp=1.0, alpha=1.0, num_classes=4)
    loss_wsld = wsld(student_logits, teacher_logits, labels)
    print(f"✓ WSLD (加权软标签): {loss_wsld.item():.4f}")

    # LSKD
    lskd = LSKDLoss()
    loss_lskd = lskd(student_logits, teacher_logits, 3)
    print(f"✓ LSKD (Logit标准化): {loss_lskd.item():.4f}")

    print()


def test_feature_losses():
    """测试特征损失函数 - 仅包含14种指定方法"""
    print("=" * 60)
    print("测试Feature-based蒸馏方法 (14种)")
    print("=" * 60)

    # 准备测试数据
    batch_size = 4
    num_classes = 4

    # AT (Attention Transfer)
    student_feat = torch.randn(batch_size, 64, 7, 7)
    teacher_feat = torch.randn(batch_size, 128, 7, 7)
    at_loss = AttentionLoss(p=2)
    loss_at = at_loss([student_feat], [teacher_feat])
    print(f"✓ AT (注意力迁移): {loss_at[0].item():.4f}")

    # FSP (Flow of Solution Procedure)
    fsp_loss = FSPLoss()
    student_layer1 = torch.randn(batch_size, 64, 7, 7)
    student_layer2 = torch.randn(batch_size, 64, 7, 7)
    teacher_layer1 = torch.randn(batch_size, 64, 7, 7)
    teacher_layer2 = torch.randn(batch_size, 64, 7, 7)
    loss_fsp = fsp_loss([student_layer1, student_layer2], [teacher_layer1, teacher_layer2])
    print(f"✓ FSP (求解流程): {sum(loss_fsp).item():.4f}")

    # SP (Similarity-Preserving)
    sp_loss = SimilarityLoss()
    loss_sp = sp_loss([student_feat], [teacher_feat])
    print(f"✓ SP (相似性保持): {loss_sp[0].item():.4f}")

    # PKT (Probabilistic Knowledge Transfer)
    pkt_loss = PKTLoss()
    s_feat = student_feat.view(batch_size, -1)
    t_feat = teacher_feat.view(batch_size, -1)
    loss_pkt = pkt_loss(s_feat, t_feat)
    print(f"✓ PKT (概率知识迁移): {loss_pkt.item():.4f}")

    # RKD (Relational Knowledge Distillation)
    rkd_loss = RKDLoss(w_d=25, w_a=50)
    loss_rkd = rkd_loss(s_feat, t_feat)
    print(f"✓ RKD (关系知识蒸馏): {loss_rkd.item():.4f}")

    # NST (Neuron Selectivity Transfer)
    nst_loss = NSTLoss()
    loss_nst = nst_loss([student_feat], [teacher_feat])
    print(f"✓ NST (神经元选择性迁移): {loss_nst[0].item():.4f}")

    # VID (Variational Information Distillation)
    vid_loss = VIDLoss(
        num_input_channels=256,
        num_mid_channel=256,
        num_target_channels=512,
        init_pred_var=5.0,
        eps=1e-5
    )
    student_feat_large = torch.randn(batch_size, 256, 14, 14)
    teacher_feat_large = torch.randn(batch_size, 512, 7, 7)
    loss_vid = vid_loss(student_feat_large, teacher_feat_large)
    print(f"✓ VID (变分信息蒸馏): {loss_vid.item():.4f}")

    # CC (Correlation Congruence)
    cc_loss = CorrelationLoss()
    feat_s = torch.randn(batch_size, 64, 7, 7)
    feat_t = torch.randn(batch_size, 64, 7, 7)
    loss_cc = cc_loss(feat_s, feat_t)
    print(f"✓ CC (相关性一致性): {loss_cc.item():.4f}")

    # ICKD (Inner Correlation KD)
    ickd_loss = ICKDLoss(student_channels=256, teacher_channels=512)
    loss_ickd = ickd_loss(student_feat_large, teacher_feat_large)
    print(f"✓ ICKD (内部相关性蒸馏): {loss_ickd.item():.4f}")

    # MGD (Masked Generative Distillation)
    mgd_loss = MGDLoss(student_channels=256, teacher_channels=512)
    loss_mgd = mgd_loss(student_feat_large, teacher_feat_large)
    print(f"✓ MGD (掩码生成蒸馏): {loss_mgd.item():.4f}")

    # SDD (Structure Decoupled Distillation)
    sdd_loss = create_distillation_loss('sdd', sdd_temperature=4.0)
    student_multi = torch.randn(batch_size, num_classes, 5)  # [B, C, N]
    teacher_multi = torch.randn(batch_size, num_classes, 5)  # [B, C, N]
    labels = torch.randint(0, num_classes, (batch_size,))
    try:
        loss_sdd = sdd_loss(student_multi, teacher_multi, labels)
        print(f"✓ SDD (结构解耦蒸馏): {loss_sdd.item():.4f}")
    except NotImplementedError as e:
        print(f"✗ SDD: {e}")

    # VkD (Variational KD)
    vkd_loss = VKDLoss(student_channels=256, teacher_channels=512)
    student_logits = torch.randn(batch_size, num_classes)
    teacher_logits = torch.randn(batch_size, num_classes)
    loss_vkd = vkd_loss(student_feat_large, teacher_feat_large, student_logits, teacher_logits)
    print(f"✓ VkD (变分蒸馏): {loss_vkd.item():.4f}")

    # DiffKD (Diffusion-based KD)
    print("\n[DiffKD (扩散蒸馏) 动态导入测试]")
    diffkd = create_distillation_loss('diffkd', student_channels=256, teacher_channels=512)
    if hasattr(diffkd.kd_loss, 'available') and diffkd.kd_loss.available:
        print("  ✓ 依赖模块已加载")
        try:
            loss_diffkd = diffkd(student_feat_large, teacher_feat_large)
            print(f"  ✓ DiffKD: {loss_diffkd.item():.4f}")
        except Exception as e:
            print(f"  ✗ 计算错误: {e}")
    else:
        print("  ✓ 依赖模块未加载 (预期行为)")
        try:
            loss = diffkd(student_feat_large, teacher_feat_large)
        except NotImplementedError as e:
            msg = str(e)
            print("  调用时返回详细错误提示:")
            lines = msg.split('\n')[:6]
            for line in lines:
                print(f"    {line}")

    # CAT_KD (Channel-Aligned Transfer KD)
    cat_kd_loss = CAT_KDLoss(student_channels=256, teacher_channels=512)
    loss_cat_kd = cat_kd_loss(student_feat_large, teacher_feat_large)
    print(f"✓ CAT_KD (通道对齐蒸馏): {loss_cat_kd.item():.4f}")

    # MKD (Masked KD - 完整实现)
    print("\n[MKD (掩码蒸馏) 完整实现测试]")
    print("  模块包含: 对齐层(Align)、空间注意力模块(SAM)、解码器(Decoder)、空间重建模块(SRM)")
    mkd_loss = create_distillation_loss('mkd', student_channels=256, teacher_channels=512,
                                       mask_ratio=0.1, depth=4)
    print(f"  ✓ MKD 创建成功")
    print(f"  ✓ Decoder 深度: {mkd_loss.kd_loss.depth}")
    try:
        # 测试 4D 输入
        loss_4d = mkd_loss(student_feat_large, teacher_feat_large)
        print(f"  ✓ MKD (4D): {loss_4d.item():.4f}")
        # 测试 3D 音频输入
        student_feat_3d = torch.randn(batch_size, 100, 256)
        teacher_feat_3d = torch.randn(batch_size, 100, 512)
        loss_3d = mkd_loss(student_feat_3d, teacher_feat_3d)
        print(f"  ✓ MKD (3D 音频): {loss_3d.item():.4f}")
    except Exception as e:
        print(f"  ✗ 计算错误: {e}")

    print()


def test_freekd():
    """测试 FreeKD (使用小波变换)"""
    print("=" * 60)
    print("测试 FreeKD (自由蒸馏)")
    print("=" * 60)

    batch_size = 4
    student_feat = torch.randn(batch_size, 256, 14, 14)
    teacher_feat = torch.randn(batch_size, 512, 7, 7)

    freekd_loss = create_distillation_loss('freekd', student_channels=256, teacher_channels=512)
    if hasattr(freekd_loss.kd_loss, 'available') and freekd_loss.kd_loss.available:
        print("  ✓ pytorch_wavelets 库已加载")
        try:
            loss = freekd_loss(student_feat, teacher_feat)
            print(f"✓ FreeKD: {loss.item():.4f}")
        except Exception as e:
            print(f"✗ 计算错误: {e}")
    else:
        print("✓ pytorch_wavelets 库未加载 (预期行为)")
        try:
            loss = freekd_loss(student_feat, teacher_feat)
        except NotImplementedError as e:
            msg = str(e)
            print("  调用时返回详细错误提示:")
            lines = msg.split('\n')[:6]
            for line in lines:
                print(f"    {line}")

    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("开始测试指定的20种蒸馏损失函数")
    print("=" * 60 + "\n")

    test_logit_losses()   # Logit-based: KD, DKD, NKD, WSLD, LSKD, FreeKD
    test_feature_losses()  # Feature-based: AT, FSP, SP, PKT, RKD, NST, VID, CC, ICKD, MGD, SDD, VkD, DiffKD, CAT_KD, MKD
    test_freekd()  # FreeKD 单独测试

    print("=" * 60)
    print("所有指定的20种蒸馏方法测试完成！")
    print("=" * 60)
    print("\n总结:")
    print("- Logit-based 方法 (6种):")
    print("  1. KD - 基础知识蒸馏")
    print("  2. DKD - 解耦知识蒸馏")
    print("  3. NKD - 负知识蒸馏")
    print("  4. WSLD - 加权软标签蒸馏")
    print("  5. LSKD - Logit标准化蒸馏")
    print("  6. FreeKD - 自由蒸馏 (使用小波变换)")
    print("\n- Feature-based 方法 (15种):")
    print("  1. AT - 注意力迁移")
    print("  2. FSP - 求解流程")
    print("  3. SP - 相似性保持")
    print("  4. PKT - 概率知识迁移")
    print("  5. RKD - 关系知识蒸馏")
    print("  6. NST - 神经元选择性迁移")
    print("  7. VID - 变分信息蒸馏")
    print("  8. CC - 相关性一致性")
    print("  9. ICKD - 内部相关性蒸馏")
    print("  10. MGD - 掩码生成蒸馏")
    print("  11. SDD - 结构解耦蒸馏")
    print("  12. VkD - 变分蒸馏")
    print("  13. DiffKD - 扩散蒸馏")
    print("  14. CAT_KD - 通道对齐蒸馏")
    print("  15. MKD - 掩码蒸馏 (完整实现)")
