import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import yaml

from models.LNN import AudioCfC
from datasets.audio_dataset import AudioDataset, validation_test_transform


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 确保数值类型正确
    config['training']['lr'] = float(config['training']['lr'])
    config['training']['weight_decay'] = float(config['training']['weight_decay'])
    
    return config


def get_inputs(input_data, data_type, device):
    """根据数据类型获取输入"""
    if '@' not in data_type:
        # 单一数据类型
        return input_data[0].to(device)
    else:
        # 多模态输入 - 返回列表
        return [data.to(device) for data in input_data]


def evaluate_model(model, data_loader, device, data_type):
    """评估模型并返回预测结果"""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, (input_data, labels) in enumerate(data_loader):
            # 使用与训练脚本相同的方式获取输入
            inputs = get_inputs(input_data, data_type, device)
            labels = labels.to(device)
            
            # 学生网络只需要 wav_s (学生音频输入)
            # 对于 wav_s@wav_t, input_data[0] 是 wav_s, input_data[1] 是 wav_t
            # 学生模型只需要 wav_s
            if '@' in data_type:
                # 多模态输入，学生模型只需要第一个模态 (wav_s)
                student_input = inputs[0]
            else:
                student_input = inputs
            
            # 前向传播
            outputs, seq_features, fl, fg, bl, bg, x_encoder = model(student_input)
            
            # 获取预测概率和预测类别
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Student Model on ShipsEar Dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to best_student.pth file')
    parser.add_argument('--config', type=str,
                        default='/media/hdd1/fubohan/Code/UATR/configs/train_distillation_shipsear.yaml',
                        help='Path to config file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ID to use')
    parser.add_argument('--dataset', type=str, default='test',
                        choices=['train', 'validation', 'test'],
                        help='Dataset to evaluate on')
    args = parser.parse_args()
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载配置
    config = load_config(args.config)
    print(f"Config loaded from: {args.config}")
    
    # 获取数据集和模型参数
    data_dir = config['dataset']['data_dir']
    data_type = config['dataset']['data_type']
    num_classes = config['model']['num_classes']
    p_encoder = config['model']['student'].get('p_encoder', 0.2)
    p_classifier = config['model']['student'].get('p_classifier', 0.3)
    
    print(f"\nDataset:")
    print(f"  Data dir: {data_dir}")
    print(f"  Data type: {data_type}")
    print(f"  Dataset to evaluate: {args.dataset}")
    print(f"\nModel:")
    print(f"  Num classes: {num_classes}")
    print(f"  p_encoder: {p_encoder}")
    print(f"  p_classifier: {p_classifier}")
    print(f"  Checkpoint: {args.checkpoint}")
    
    # 创建数据集
    print(f"\nLoading {args.dataset} dataset...")
    test_dataset = AudioDataset(
        data_dir=data_dir,
        data_flag=args.dataset,
        data_type=data_type,
        transform=validation_test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"  Dataset size: {len(test_dataset)}")
    
    # 创建模型
    print(f"\nCreating student model...")
    model = AudioCfC(
        num_classes=num_classes,
        p_encoder=p_encoder,
        p_classifier=p_classifier
    ).to(device)
    
    # 加载模型权重
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 检查checkpoint的格式
    if 'student_state_dict' in checkpoint:
        state_dict = checkpoint['student_state_dict']
        if 'best_acc' in checkpoint:
            print(f"  Best accuracy during training: {checkpoint['best_acc']:.2f}%")
    else:
        state_dict = checkpoint
        print("  Warning: Checkpoint does not contain 'student_state_dict', using whole checkpoint")
    
    model.load_state_dict(state_dict, strict=False)
    print("  Checkpoint loaded successfully!")
    
    # 评估模型
    print(f"\nEvaluating model on {args.dataset} set...")
    labels, preds, probs = evaluate_model(model, test_loader, device, data_type)
    
    # 计算指标
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Accuracy
    acc = accuracy_score(labels, preds) * 100
    print(f"\nAccuracy: {acc:.4f}%")
    
    # F1 Score
    f1_macro = f1_score(labels, preds, average='macro') * 100
    f1_micro = f1_score(labels, preds, average='micro') * 100
    f1_weighted = f1_score(labels, preds, average='weighted') * 100
    print(f"\nF1 Score:")
    print(f"  Macro:    {f1_macro:.4f}%")
    print(f"  Micro:    {f1_micro:.4f}%")
    print(f"  Weighted:  {f1_weighted:.4f}%")
    
    # AUC (for multi-class)
    try:
        if num_classes > 2:
            # Multi-class AUC
            auc_ovo = roc_auc_score(labels, probs, multi_class='ovo') * 100
            auc_ovr = roc_auc_score(labels, probs, multi_class='ovr') * 100
            print(f"\nAUC Score:")
            print(f"  OVO (One-vs-One):   {auc_ovo:.4f}%")
            print(f"  OVR (One-vs-Rest):  {auc_ovr:.4f}%")
        else:
            # Binary AUC
            auc = roc_auc_score(labels, probs[:, 1]) * 100
            print(f"\nAUC Score: {auc:.4f}%")
    except Exception as e:
        print(f"\nWarning: Could not compute AUC - {e}")
    
    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Classification Report
    print(f"\nClassification Report:")
    if num_classes == 5:
        target_names = ['A', 'D', 'C', 'B', 'E']
    elif num_classes == 4:
        target_names = ['passengership', 'tanker', 'cargo', 'tug']
    else:
        target_names = [f'Class {i}' for i in range(num_classes)]
    
    report = classification_report(labels, preds, target_names=target_names, digits=4)
    print(report)
    
    # Per-class metrics
    print("\nPer-class Metrics:")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 65)
    
    report_dict = classification_report(labels, preds, target_names=target_names, output_dict=True)
    for class_name in target_names:
        metrics = report_dict[class_name]
        print(f"{class_name:<15} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1-score']:<12.4f} {int(metrics['support']):<10}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETED")
    print("="*60)
    
    # 保存结果到文件
    output_dir = os.path.dirname(args.checkpoint)
    results_file = os.path.join(output_dir, f'evaluation_results_{args.dataset}.txt')
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"EVALUATION RESULTS - {args.dataset.upper()} SET\n")
        f.write("="*60 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: {data_dir}\n")
        f.write(f"Dataset split: {args.dataset}\n")
        f.write(f"Number of samples: {len(test_dataset)}\n\n")
        
        f.write(f"Accuracy: {acc:.4f}%\n\n")
        
        f.write(f"F1 Score:\n")
        f.write(f"  Macro:    {f1_macro:.4f}%\n")
        f.write(f"  Micro:    {f1_micro:.4f}%\n")
        f.write(f"  Weighted:  {f1_weighted:.4f}%\n\n")
        
        if num_classes > 2:
            f.write(f"AUC Score:\n")
            f.write(f"  OVO (One-vs-One):   {auc_ovo:.4f}%\n")
            f.write(f"  OVR (One-vs-Rest):  {auc_ovr:.4f}%\n\n")
        else:
            f.write(f"AUC Score: {auc:.4f}%\n\n")
        
        f.write(f"Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        
        f.write(f"Classification Report:\n")
        f.write(report + "\n")
    
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
