import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import numpy as np
import pandas as pd
from datetime import datetime
from datasets.audio_dataset import AudioDataset, validation_test_transform
from models.Audio_TeacherNet import build_Audio_TeacherNet


def load_model(model_path, num_classes, input_dim, device):
    """加载模型和权重"""
    # 初始化模型（build_Audio_TeacherNet会自动加载checkpoint）
    model = build_Audio_TeacherNet(num_classes=num_classes, checkpoint_path=model_path)
    
    model = model.to(device)
    model.eval()
    
    print(f'Model loaded from: {model_path}')
    
    return model


def get_inputs(input_data, data_type, device):
    """根据数据类型获取输入（与audio_dataset.py保持一致）"""
    if '@' not in data_type:
        # 单一数据类型 - 直接返回第一个元素
        return input_data[0].to(device)
    else:
        # 多模态输入
        return [data.to(device) for data in input_data]


def validate(model, data_loader, device, data_type):
    """验证模型"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_outputs = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (input_data, labels) in enumerate(data_loader):
            # 获取输入（与audio_dataset.py的__getitem__返回格式一致）
            inputs = get_inputs(input_data, data_type, device)
            labels = labels.to(device)
            
            # 前向传播（Audio_TeacherNet返回4个值：output, extract_features, last_hidden_state, all_hidden_states）
            outputs, _, _, _ = model(inputs)
            
            # 获取预测
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Processing batch [{batch_idx + 1}/{len(data_loader)}]')
    
    # 计算准确率
    accuracy = 100. * correct / total
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_outputs = np.array(all_outputs)
    
    return accuracy, all_preds, all_labels, all_outputs


def print_results(accuracy, all_preds, all_labels, all_outputs, class_names=None):
    """打印评估结果"""
    print('\n' + '='*80)
    print('VALIDATION RESULTS')
    print('='*80)
    
    print(f'\nAccuracy: {accuracy:.4f}%')
    
    # F1 Score
    f1_macro = f1_score(all_labels, all_preds, average='macro') * 100
    f1_micro = f1_score(all_labels, all_preds, average='micro') * 100
    f1_weighted = f1_score(all_labels, all_preds, average='weighted') * 100
    
    print(f'\nF1 Score:')
    print(f'  Macro:   {f1_macro:.4f}%')
    print(f'  Micro:   {f1_micro:.4f}%')
    print(f'  Weighted: {f1_weighted:.4f}%')
    
    # AUC (多分类)
    try:
        auc_ovo = roc_auc_score(all_labels, all_outputs, multi_class='ovo') * 100
        auc_ovr = roc_auc_score(all_labels, all_outputs, multi_class='ovr') * 100
        print(f'\nAUC Score:')
        print(f'  OVO (One-vs-One):  {auc_ovo:.4f}%')
        print(f'  OVR (One-vs-Rest): {auc_ovr:.4f}%')
    except Exception as e:
        print(f'\nAUC Score: Unable to calculate ({str(e)})')
    
    # 混淆矩阵
    print(f'\nConfusion Matrix:')
    print('-'*80)
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    # 分类报告
    print(f'\nClassification Report:')
    print('-'*80)
    if class_names is not None:
        print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    else:
        print(classification_report(all_labels, all_preds, digits=4))
    
    # 每个类别的准确率
    print(f'\nPer-class Accuracy:')
    print('-'*80)
    for i in range(len(cm)):
        class_total = cm[i].sum()
        class_correct = cm[i][i]
        class_acc = 100. * class_correct / class_total if class_total > 0 else 0
        class_name = class_names[i] if class_names is not None else f'Class {i}'
        print(f'{class_name}: {class_acc:.2f}% ({class_correct}/{class_total})')
    
    print('='*80)


def get_correct_class_names(data_dir, data_flag):
    """
    获取正确的类别名称
    注意：这里与audio_dataset.py的逻辑保持一致
    但使用正确的标签顺序
    """
    wav_dir = os.path.join(data_dir, data_flag, 'wav')
    
    # 获取实际的文件夹名称
    actual_category_names = sorted(os.listdir(wav_dir))
    
    # 根据audio_dataset.py的逻辑返回类别名称
    # 但确保标签顺序是正确的（与实际文件夹顺序一致）
    if len(actual_category_names) == 5:
        # ShipsEar: 根据audio_dataset.py，会强制改为 ['A', 'D', 'C', 'B', 'E']
        # 但这是错误的！我们应该返回正确的顺序 ['A', 'B', 'C', 'D', 'E']
        # 标签0对应A，标签1对应B，标签2对应C，标签3对应D，标签4对应E
        return ['A', 'B', 'C', 'D', 'E']
    elif len(actual_category_names) == 4:
        # DeepShip: 根据audio_dataset.py，会强制改为 ['passengership', 'tanker', 'cargo', 'tug']
        # 这个顺序是正确的
        return ['passengership', 'tanker', 'cargo', 'tug']
    else:
        # 其他情况，返回实际的文件夹名称（已排序）
        return actual_category_names


def print_results_summary(result):
    """打印简化的结果摘要"""
    print('\n' + '='*80)
    print('VALIDATION RESULTS')
    print('='*80)
    print(f'Accuracy: {result["accuracy"]:.4f}%')
    print(f'\nF1 Score:')
    print(f'  Macro:   {result["f1_macro"]:.4f}%')
    print(f'  Micro:   {result["f1_micro"]:.4f}%')
    print(f'  Weighted: {result["f1_weighted"]:.4f}%')
    if result["auc_ovo"] is not None:
        print(f'\nAUC Score:')
        print(f'  OVO (One-vs-One):  {result["auc_ovo"]:.4f}%')
        print(f'  OVR (One-vs-Rest): {result["auc_ovr"]:.4f}%')
    print(f'\nPer-class Accuracy:')
    for i, acc in enumerate(result["per_class_accuracy"]):
        print(f'  Class {i}: {acc:.2f}%')
    print('='*80)


def save_results_to_csv(results, csv_path):
    """将验证结果保存到CSV文件"""
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f'\n✓ Results saved to: {csv_path}')


def evaluate_model(model_path, num_classes, data_dir, data_type, data_flag, batch_size, num_workers, gpu):
    """评估单个模型在单个数据集上的性能"""
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu}')
        print(f'Using device: cuda:{gpu}')
    else:
        device = torch.device('cpu')
        print('Using device: cpu')
    
    # 获取正确的类别名称
    class_names = get_correct_class_names(data_dir, data_flag)
    
    # 加载模型
    model = load_model(model_path, num_classes, 1, device)
    
    # 创建数据集
    dataset = AudioDataset(
        data_dir=data_dir,
        data_flag=data_flag,
        data_type=data_type,
        transform=validation_test_transform
    )
    
    # 数据加载器
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f'Dataset: {data_flag}')
    print(f'Dataset size: {len(dataset)}')
    print(f'Classes: {class_names}\n')
    
    # 验证
    accuracy, all_preds, all_labels, all_outputs = validate(
        model, data_loader, device, data_type
    )
    
    # 计算指标
    f1_macro = f1_score(all_labels, all_preds, average='macro') * 100
    f1_micro = f1_score(all_labels, all_preds, average='micro') * 100
    f1_weighted = f1_score(all_labels, all_preds, average='weighted') * 100
    
    try:
        auc_ovo = roc_auc_score(all_labels, all_outputs, multi_class='ovo') * 100
        auc_ovr = roc_auc_score(all_labels, all_outputs, multi_class='ovr') * 100
    except:
        auc_ovo = None
        auc_ovr = None
    
    # 计算每个类别的准确率
    cm = confusion_matrix(all_labels, all_preds)
    per_class_acc = []
    for i in range(len(cm)):
        class_total = cm[i].sum()
        class_correct = cm[i][i]
        class_acc = 100. * class_correct / class_total if class_total > 0 else 0
        per_class_acc.append(class_acc)
    
    return {
        'model_path': model_path,
        'dataset': os.path.basename(data_dir),
        'data_flag': data_flag,
        'num_samples': len(dataset),
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'auc_ovo': auc_ovo,
        'auc_ovr': auc_ovr,
        'per_class_accuracy': per_class_acc
    }
def main():
    parser = argparse.ArgumentParser(description='Validation for Audio Classification')
    
    # 数据集参数
    parser.add_argument('--data_type', type=str, default='wav_t',
                        help='数据类型: wav_s / wav_t / mel / cqt / wav_s@wav_t@mel@cqt')
    
    # 其他参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--gpu', type=int, default=4,
                        help='使用的GPU ID')
    parser.add_argument('--output_csv', type=str, default='eval_teacher_acc_auc_f1_results.csv',
                        help='结果保存的CSV文件路径')
    
    args = parser.parse_args()
    
    # 定义要评估的模型和数据集配置
    evaluations = [
        # ShipsEar
        {
            'model_path': '/media/hdd1/fubohan/Code/UATR/models/Audio_Teacher_ShipsEar_622/checkpoints/Student.pth',
            'num_classes': 5,
            'data_dir': '/media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622',
            'data_flag': 'validation'
        },
        {
            'model_path': '/media/hdd1/fubohan/Code/UATR/models/Audio_Teacher_ShipsEar_622/checkpoints/Student.pth',
            'num_classes': 5,
            'data_dir': '/media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622',
            'data_flag': 'test'
        },
        # DeepShip
        {
            'model_path': '/media/hdd1/fubohan/Code/UATR/models/Audio_Teacher_DeepShip_622/checkpoints/Student.pth',
            'num_classes': 4,
            'data_dir': '/media/hdd1/chuxiaohui/AI4Ocean_UATR/DeepShip_622',
            'data_flag': 'validation'
        },
        {
            'model_path': '/media/hdd1/fubohan/Code/UATR/models/Audio_Teacher_DeepShip_622/checkpoints/Student.pth',
            'num_classes': 4,
            'data_dir': '/media/hdd1/chuxiaohui/AI4Ocean_UATR/DeepShip_622',
            'data_flag': 'test'
        },
    ]
    
    # 存储所有结果
    all_results = []
    
    # 逐个评估
    for i, eval_config in enumerate(evaluations, 1):
        print('\n' + '='*80)
        print(f'EVALUATION {i}/{len(evaluations)}')
        print('='*80)
        print(f'Model: {os.path.basename(eval_config["model_path"])}')
        print(f'Dataset: {eval_config["data_dir"]}')
        print(f'Data flag: {eval_config["data_flag"]}')
        print('='*80 + '\n')
        
        # 评估模型
        result = evaluate_model(
            model_path=eval_config['model_path'],
            num_classes=eval_config['num_classes'],
            data_dir=eval_config['data_dir'],
            data_type=args.data_type,
            data_flag=eval_config['data_flag'],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            gpu=args.gpu
        )
        
        # 打印结果
        print_results_summary(result)
        
        all_results.append(result)
    
    # 保存所有结果到CSV
    save_results_to_csv(all_results, args.output_csv)
    
    # 打印汇总表格
    print('\n' + '='*80)
    print('SUMMARY TABLE')
    print('='*80)
    summary_df = pd.DataFrame(all_results)
    display_cols = ['model_path', 'dataset', 'data_flag', 'accuracy', 'f1_macro', 'f1_weighted', 'auc_ovo', 'auc_ovr']
    print(summary_df[display_cols].to_string(index=False))
    print('='*80)


if __name__ == '__main__':
    main()
