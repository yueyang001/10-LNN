"""
模型轻量化评估脚本 - 实验四
=====================================

【使用方法】
1. 从项目根目录运行：
   cd ../../.. && python experiments/model_lightweighting/evaluate_model.py

2. 自定义输入形状：
   - 修改 __main__ 中的 input_shape=(1, 1, 48000)
   - 默认：48000 表示单通道、48kHz采样率、1秒音频

3. 自定义输出路径：
   - 修改 save_results_to_csv() 中的 save_path 参数

【输出内容】
- 模型参数量（百万）
- 模型大小（MB）
- 计算量（MFlops）
- 推理延迟（毫秒）

【依赖库】
- torch, fvcore (或 ptflops), models.LNN

【输出格式】
Model Parameters(MB)\\Model Size(MB)\\Flops(MFlops)\\Latency(Millisecond)
"""

import os
import sys
import time
import torch
import warnings

warnings.filterwarnings('ignore')

# 添加项目根路径以用于导入
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ==========================================
# 1. 导入模型
# ==========================================
try:
    from models.LNN import AudioCfC
except ImportError:
    print("[错误] 未找到 models.LNN 模块。请确认 models/LNN.py 文件存在。")
    exit()

# ==========================================
# 2. 定义评估函数
# ==========================================
 
def evaluate_model(model, input_shape=(1, 1, 48000), device_id=0):
    """
    评估模型的参数量、计算量和推理延迟
    """

    print("\n" + "="*60)
    print(" 开始模型轻量化评估")
    print("="*60)

    # 设置计算设备
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"[计算设备] 使用: {device}")

    model = model.to(device)
    model.eval()

    # --- 1. 计算参数量与模型大小 ---
    total_params = sum(p.numel() for p in model.parameters())

    # 修正：区分参数量(单位M)和模型大小(单位MB)
    num_params_m = total_params / 1e6            # 参数量（百万）
    model_size_mb = total_params * 4 / (1024 * 1024) # 模型大小（MB，float32 = 4字节）

    print(f"\n[参数量统计]")
    print(f"   参数量: {num_params_m:.2f} M")
    print(f"   模型大小: {model_size_mb:.2f} MB")

    # --- 2. 计算计算量 ---
    flops_m = 0.0

    print(f"\n[计算量统计]")

    dummy_input = torch.randn(*input_shape).to(device)

    try:
        from fvcore.nn import FlopCountAnalysis
        flops_counter = FlopCountAnalysis(model, dummy_input)
        flops_total = flops_counter.total()
        flops_m = flops_total / 1e6
        print(f"   [fvcore]  FLOPs: {flops_m:.2f} M")

    except ImportError:
        print("   [警告] fvcore 未安装，尝试使用 ptflops...")
        try:
            from ptflops import get_model_complexity_info
            input_res_pt = input_shape[1:]
            macs_val, _ = get_model_complexity_info(model, input_res_pt, as_strings=False, print_per_layer_stat=False, verbose=False)
            if macs_val is not None:
                flops_m = (macs_val * 2) / 1e6
                print(f"   [ptflops] FLOPs: {flops_m:.2f} M")
        except ImportError:
            print("   [错误] 无法计算 FLOPs，请安装 fvcore 或 ptflops")

    # --- 3. 测量推理延迟 ---
    print(f"\n[推理延迟测试]")

    print("   正在预热...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    print("   正在测量...")
    num_runs = 100
    total_time = 0.0

    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()

            start_time = time.time()
            _ = model(dummy_input)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            end_time = time.time()
            total_time += (end_time - start_time)

    avg_latency_ms = (total_time / num_runs) * 1000
    print(f"   平均延迟: {avg_latency_ms:.2f} ms")

    # --- 4. 汇总结果 ---
    print("\n" + "="*60)
    print(" 评估完成 - 实验四结果汇总")
    print("="*60)
    print(f"{'方法':<15} {'参数量(M)':<15} {'大小(MB)':<15} {'计算量(M)':<15} {'延迟(ms)':<15}")
    print("-" * 75)
    print(f"{'LNN (Ours)':<15} {num_params_m:<15.2f} {model_size_mb:<15.2f} {flops_m:<15.2f} {avg_latency_ms:<15.2f}")
    print("="*60)

    # 按指定格式输出：Model Parameters(MB)\Model Size(MB)\Flops(MFlops)\Latency(Millisecond)
    print("\n[标准输出格式]")
    print(f"Model Parameters(MB)\\Model Size(MB)\\Flops(MFlops)\\Latency(Millisecond)")
    print(f"{num_params_m:.2f}\\{model_size_mb:.2f}\\{flops_m:.2f}\\{avg_latency_ms:.2f}")

    return {
        "num_params_m": num_params_m,   # 返回参数量
        "model_size_mb": model_size_mb, # 返回模型大小
        "flops": flops_m,               # 返回计算量
        "latency": avg_latency_ms       # 返回推理延迟
    }

def save_results_to_csv(results, save_path="results/ablation_shipsear/experiment4_results.csv"):
    """保存结果到 CSV 文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    import csv
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['指标', '数值'])
        # 分开写入
        writer.writerow(['模型参数量 (M)', f"{results['num_params_m']:.2f}"])     # 单位：百万
        writer.writerow(['模型大小 (MB)', f"{results['model_size_mb']:.2f}"])      # 单位：MB
        writer.writerow(['计算量 (MFLOPs)', f"{results['flops']:.2f}"])
        writer.writerow(['推理延迟 (ms)', f"{results['latency']:.2f}"])

    print(f"[成功] 结果已保存至: {save_path}")

# ==========================================
# 3. 主程序入口
# ==========================================

if __name__ == "__main__":
    print("正在初始化模型...")
    model = AudioCfC(num_classes=4)
    results = evaluate_model(model, input_shape=(1, 1, 48000), device_id=0)
    save_results_to_csv(results)