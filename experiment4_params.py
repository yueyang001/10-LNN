import os
import time
import torch
import warnings
 
warnings.filterwarnings('ignore')
 
# ==========================================
# 1. 导入模型
# ==========================================
try:
    from models.LNN import AudioCfC
except ImportError:
    print("❌ 错误：未找到 models.LNN 模块。请确认 models/LNN.py 文件存在。")
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
    
    # 设置设备
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备: {device}")
    
    model = model.to(device)
    model.eval()
    
    # --- 1. 计算参数量与模型大小 ---
    total_params = sum(p.numel() for p in model.parameters())
    
    # 🔧 修正：区分参数量(单位M)和模型大小(单位MB)
    num_params_m = total_params / 1e6            # 参数量
    model_size_mb = total_params * 4 / (1024 * 1024) # 模型大小 (float32 = 4 bytes)
    
    print(f"\n📊 [参数量统计]")
    print(f"   参数量: {num_params_m:.2f} M")
    print(f"   模型大小: {model_size_mb:.2f} MB")
 
    # --- 2. 计算计算量 ---
    flops_m = 0.0
    
    print(f"\n⚡ [计算量统计]")
    
    dummy_input = torch.randn(*input_shape).to(device)
    
    try:
        from fvcore.nn import FlopCountAnalysis
        flops_counter = FlopCountAnalysis(model, dummy_input)
        flops_total = flops_counter.total()
        flops_m = flops_total / 1e6
        print(f"   [fvcore]  FLOPs: {flops_m:.2f} M")
        
    except ImportError:
        print("   ⚠️ fvcore 未安装，尝试使用 ptflops...")
        try:
            from ptflops import get_model_complexity_info
            input_res_pt = input_shape[1:] 
            macs_val, _ = get_model_complexity_info(model, input_res_pt, as_strings=False, print_per_layer_stat=False, verbose=False)
            if macs_val is not None:
                flops_m = (macs_val * 2) / 1e6
                print(f"   [ptflops] FLOPs: {flops_m:.2f} M")
        except ImportError:
            print("   ❌ 无法计算 FLOPs，请安装 fvcore 或 ptflops")
 
    # --- 3. 测量推理延迟 ---
    print(f"\n⏱️  [推理延迟测试]")
    
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
    # 🔧 修正：表格列名与实验要求对齐
    print(f"{'Method':<15} {'Params(M)':<15} {'Size(MB)':<15} {'FLOPs(M)':<15} {'Latency(ms)':<15}")
    print("-" * 75)
    print(f"{'LNN (Ours)':<15} {num_params_m:<15.2f} {model_size_mb:<15.2f} {flops_m:<15.2f} {avg_latency_ms:<15.2f}")
    print("="*60)
    
    return {
        "num_params_m": num_params_m,   # 返回参数量
        "model_size_mb": model_size_mb, # 返回模型大小
        "flops": flops_m,
        "latency": avg_latency_ms
    }
 
def save_results_to_csv(results, save_path="results/ablation_shipsear/experiment4_results.csv"):
    """保存结果到 CSV"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    import csv
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        # 🔧 修正：分开写入
        writer.writerow(['Model Parameters (M)', f"{results['num_params_m']:.2f}"]) # 单位 Million
        writer.writerow(['Model Size (MB)', f"{results['model_size_mb']:.2f}"])     # 单位 MB
        writer.writerow(['FLOPs (MFLOPS)', f"{results['flops']:.2f}"])
        writer.writerow(['Latency (ms)', f"{results['latency']:.2f}"])
    
    print(f"✅ 结果已保存至: {save_path}")
 
# ==========================================
# 3. 主程序入口
# ==========================================
 
if __name__ == "__main__":
    print("正在初始化模型...")
    model = AudioCfC(num_classes=4)
    results = evaluate_model(model, input_shape=(1, 1, 48000), device_id=0)
    save_results_to_csv(results)