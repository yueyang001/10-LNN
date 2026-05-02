import os
import subprocess
import glob
import numpy as np

# 配置
base_dir = '/media/hdd1/fubohan/Code/UATR/checkpoints/ablation_deepship'
config_file = '/media/hdd1/fubohan/Code/UATR/configs/train_distillation_deepship.yaml'
batch_size = 32
gpu_id = '0'
dataset = 'test'  # 可以是 'train', 'validation', 'test'
conda_env = 'UATR'  # conda 环境名称

# 找到所有包含 best_student.pth 的目录
print("Searching for models in:", base_dir)
print()

model_dirs = []
for item in os.listdir(base_dir):
    item_path = os.path.join(base_dir, item)
    if os.path.isdir(item_path):
        best_student_path = os.path.join(item_path, 'best_student.pth')
        if os.path.exists(best_student_path):
            model_dirs.append((item, best_student_path))

if not model_dirs:
    print("No best_student.pth files found!")
    exit(1)

print(f"Found {len(model_dirs)} model(s) to evaluate:")
for dir_name, _ in model_dirs:
    print(f"  - {dir_name}")
print()

# 创建结果汇总文件
summary_file = os.path.join(base_dir, f'evaluation_summary_{dataset}.txt')
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("BATCH EVALUATION SUMMARY - ABILATION DEEPSHIP\n")
    f.write("="*80 + "\n\n")
    f.write(f"Config: {config_file}\n")
    f.write(f"Dataset: {dataset}\n")
    f.write(f"Batch size: {batch_size}\n")
    f.write(f"GPU: {gpu_id}\n")
    f.write(f"Total models: {len(model_dirs)}\n\n")
    f.write("-"*80 + "\n\n")

results = []

# 逐个评估模型
for idx, (dir_name, checkpoint_path) in enumerate(model_dirs, 1):
    print(f"\n{'='*60}")
    print(f"[{idx}/{len(model_dirs)}] Evaluating: {dir_name}")
    print(f"{'='*60}")
    
    # 构建命令 - 使用 UATR conda 环境
    cmd = f"""
    source /home/fubohan/anaconda3/etc/profile.d/conda.sh && \
    conda activate {conda_env} && \
    python /media/hdd1/fubohan/Code/UATR/evaluate_student_deepship.py \
        --checkpoint {checkpoint_path} \
        --config {config_file} \
        --batch_size {batch_size} \
        --gpu {gpu_id} \
        --dataset {dataset}
    """
    
    print(f"Running command: {' '.join(cmd)}")
    print()
    
    # 运行评估脚本
    try:
        result = subprocess.run(
            cmd,
            cwd='/media/hdd1/fubohan/Code/UATR',
            shell=True,
            capture_output=True,
            text=True,
            timeout=600  # 10分钟超时
        )
        
        # 输出结果
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # 解析结果
        acc = None
        f1_macro = None
        f1_micro = None
        auc_ovo = None
        auc_ovr = None
        
        for line in result.stdout.split('\n'):
            if 'Accuracy:' in line and '%' in line:
                acc = float(line.split('Accuracy:')[1].strip().split('%')[0])
            elif 'Macro:' in line and 'F1 Score' not in line:
                f1_macro = float(line.split('Macro:')[1].strip().split('%')[0])
            elif 'Micro:' in line and 'F1 Score' not in line:
                f1_micro = float(line.split('Micro:')[1].strip().split('%')[0])
            elif 'OVO' in line:
                auc_ovo = float(line.split('OVO')[1].split('%')[0].strip().split(':')[1].strip())
            elif 'OVR' in line:
                auc_ovr = float(line.split('OVR')[1].split('%')[0].strip().split(':')[1].strip())
        
        # 保存结果
        results.append({
            'dir_name': dir_name,
            'checkpoint': checkpoint_path,
            'acc': acc,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'auc_ovo': auc_ovo,
            'auc_ovr': auc_ovr,
            'success': result.returncode == 0
        })
        
        # 写入汇总文件
        with open(summary_file, 'a', encoding='utf-8') as f:
            f.write(f"[{idx}/{len(model_dirs)}] {dir_name}\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            if acc is not None:
                f.write(f"Accuracy: {acc:.4f}%\n")
            if f1_macro is not None:
                f.write(f"F1-Macro: {f1_macro:.4f}%\n")
            if f1_micro is not None:
                f.write(f"F1-Micro: {f1_micro:.4f}%\n")
            if auc_ovo is not None:
                f.write(f"AUC-OVO: {auc_ovo:.4f}%\n")
            if auc_ovr is not None:
                f.write(f"AUC-OVR: {auc_ovr:.4f}%\n")
            f.write(f"Status: {'SUCCESS' if result.returncode == 0 else 'FAILED'}\n")
            f.write("-"*80 + "\n\n")
        
    except subprocess.TimeoutExpired:
        print(f"ERROR: Evaluation timed out for {dir_name}")
        results.append({
            'dir_name': dir_name,
            'checkpoint': checkpoint_path,
            'acc': None,
            'f1_macro': None,
            'f1_micro': None,
            'auc_ovo': None,
            'auc_ovr': None,
            'success': False
        })
        
        with open(summary_file, 'a', encoding='utf-8') as f:
            f.write(f"[{idx}/{len(model_dirs)}] {dir_name}\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Status: TIMEOUT\n")
            f.write("-"*80 + "\n\n")
    
    except Exception as e:
        print(f"ERROR: {e}")
        results.append({
            'dir_name': dir_name,
            'checkpoint': checkpoint_path,
            'acc': None,
            'f1_macro': None,
            'f1_micro': None,
            'auc_ovo': None,
            'auc_ovr': None,
            'success': False,
            'error': str(e)
        })
        
        with open(summary_file, 'a', encoding='utf-8') as f:
            f.write(f"[{idx}/{len(model_dirs)}] {dir_name}\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Status: ERROR - {e}\n")
            f.write("-"*80 + "\n\n")

# 打印最终汇总
print("\n" + "="*80)
print("BATCH EVALUATION COMPLETED")
print("="*80)
print(f"\nSummary saved to: {summary_file}")

# 打印排名
print("\n" + "="*80)
print("RANKING BY ACCURACY")
print("="*80)

successful_results = [r for r in results if r['success'] and r['acc'] is not None]
successful_results.sort(key=lambda x: x['acc'], reverse=True)

if successful_results:
    print(f"\n{'Rank':<6} {'Model':<50} {'Acc(%)':<10} {'F1-Macro(%)':<12} {'AUC-OVO(%)':<12}")
    print("-"*90)
    for rank, result in enumerate(successful_results, 1):
        name = result['dir_name'][:48] + '..' if len(result['dir_name']) > 48 else result['dir_name']
        print(f"{rank:<6} {name:<50} {result['acc']:<10.4f} {result['f1_macro']:<12.4f} {result['auc_ovo']:<12.4f}")
else:
    print("\nNo successful evaluations!")

# 统计
print(f"\nStatistics:")
print(f"  Total models: {len(model_dirs)}")
print(f"  Successful: {len(successful_results)}")
print(f"  Failed: {len(results) - len(successful_results)}")

if successful_results:
    accs = [r['acc'] for r in successful_results]
    print(f"\nAccuracy Statistics:")
    print(f"  Best: {max(accs):.4f}%")
    print(f"  Worst: {min(accs):.4f}%")
    print(f"  Average: {np.mean(accs):.4f}%")
    print(f"  Std: {np.std(accs):.4f}%")
