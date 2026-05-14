import re
import matplotlib.pyplot as plt
import os
import sys
# python scripts/analyze_log.py <shipsear_pth_path> <deepship_pth_path>
# 脚本会自动从 pth 文件所在目录查找对应的 log 文件
# ========= 输入 pth 文件路径 =========
if len(sys.argv) < 3:
    print("Usage: python analyze_log.py <shipsear_pth_path> <deepship_pth_path>")
    print("Example: python analyze_log.py checkpoints/ablation_shipsear/.../model.pth checkpoints/ablation_deepship/.../model.pth")
    sys.exit(1)

shipsear_pth = sys.argv[1]
deepship_pth = sys.argv[2]

# ========= 从 pth 路径推导 log 文件路径 =========
def get_log_from_pth(pth_path):
    """Extract log file from pth path."""
    dir_path = os.path.dirname(pth_path)
    log_files = [f for f in os.listdir(dir_path) if f.startswith('train_') and f.endswith('.log')]
    if log_files:
        return os.path.join(dir_path, sorted(log_files)[-1])
    return None

shipsear_log = get_log_from_pth(shipsear_pth)
deepship_log = get_log_from_pth(deepship_pth)

if not shipsear_log or not deepship_log:
    print("Error: Could not find log files")
    sys.exit(1)

print(f"ShipSEAR log: {shipsear_log}")
print(f"DeepSHIP log: {deepship_log}")

# ========= 正则：严格匹配 log =========
epoch_summary_pattern = re.compile(
    r"Epoch\s*\[(\d+)/\d+\]\s*"
    r"Train Loss:\s*([-0-9.]+),\s*Train Acc:\s*([0-9.]+)%?,\s*"
    r"Val Loss:\s*([-0-9.]+),\s*Val Acc:\s*([0-9.]+)%?"
)

best_acc_pattern = re.compile(r"Best accuracy:\s*([0-9.]+)%")

def parse_log(log_path):
    """Parse log file and return epochs, train_acc, val_acc."""
    epochs = []
    train_acc = []
    val_acc = []
    best_acc = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = epoch_summary_pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_acc.append(float(m.group(3)))
                val_acc.append(float(m.group(5)))

            b = best_acc_pattern.search(line)
            if b:
                best_acc = float(b.group(1))

    return epochs, train_acc, val_acc, best_acc

# ========= 读 log =========
shipsear_epochs, shipsear_train_acc, shipsear_val_acc, shipsear_best = parse_log(shipsear_log)
deepship_epochs, deepship_train_acc, deepship_val_acc, deepship_best = parse_log(deepship_log)

# ========= 打印统计 =========
print("\n===== ShipSEAR Statistics =====")
print(f"Epochs parsed: {len(shipsear_epochs)}")
print(f"Best Val Acc: {shipsear_best}")
print(f"Final Train Acc: {shipsear_train_acc[-1]:.2f}%")
print(f"Final Val Acc: {shipsear_val_acc[-1]:.2f}%")

print("\n===== DeepSHIP Statistics =====")
print(f"Epochs parsed: {len(deepship_epochs)}")
print(f"Best Val Acc: {deepship_best}")
print(f"Final Train Acc: {deepship_train_acc[-1]:.2f}%")
print(f"Final Val Acc: {deepship_val_acc[-1]:.2f}%")

# ========= 确保输出目录存在 =========
os.makedirs("results/overfitting_analysis", exist_ok=True)

# ========= 配色方案 =========
train_color = "#FF6B6B"  # 红色
val_color = "#4ECDC4"    # 青绿色

# ========= 画图 =========
plt.figure(figsize=(14, 5))

# 左边：ShipSEAR ACC Curve
plt.subplot(1, 2, 1)
plt.plot(shipsear_epochs, shipsear_train_acc, label="Training Acc", color=train_color, linewidth=2)
plt.plot(shipsear_epochs, shipsear_val_acc, label="Validation Acc", color=val_color, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("ShipsEar Dataset - ACC Curve")
plt.legend()
plt.text(0.5, -0.25, '(a)', transform=plt.gca().transAxes, 
         ha='center', va='top', fontsize=14)

# 右边：DeepSHIP ACC Curve
plt.subplot(1, 2, 2)
plt.plot(deepship_epochs, deepship_train_acc, label="Training Acc", color=train_color, linewidth=2)
plt.plot(deepship_epochs, deepship_val_acc, label="Validation Acc", color=val_color, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("DeepShip Dataset - ACC Curve")
plt.legend()
plt.text(0.5, -0.2, '(b)', transform=plt.gca().transAxes, 
         ha='center', va='top', fontsize=14)

plt.tight_layout()
output_path = "results/overfitting_analysis/acc_curve_comparison.png"
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"\nSaved: {output_path}")
plt.close()

