import re
import matplotlib.pyplot as plt
import os

log_path = "checkpoints/ShipsEar/dis_0204_0/train_20260204_154847.log"

# ========= 正则：严格匹配你的 log =========
epoch_summary_pattern = re.compile(
    r"Epoch\s*\[(\d+)/\d+\]\s*"
    r"Train Loss:\s*([-0-9.]+),\s*Train Acc:\s*([0-9.]+)%?,\s*"
    r"Val Loss:\s*([-0-9.]+),\s*Val Acc:\s*([0-9.]+)%?"
)

best_acc_pattern = re.compile(r"Best accuracy:\s*([0-9.]+)%")

# ========= 数据 =========
epochs = []
train_loss = []
train_acc = []
val_loss = []
val_acc = []
best_acc = None

# ========= 读 log =========
with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = epoch_summary_pattern.search(line)
        if m:
            epochs.append(int(m.group(1)))
            train_loss.append(float(m.group(2)))
            train_acc.append(float(m.group(3)))
            val_loss.append(float(m.group(4)))
            val_acc.append(float(m.group(5)))

        b = best_acc_pattern.search(line)
        if b:
            best_acc = float(b.group(1))

# ========= 打印统计 =========
print("===== Epoch-level Statistics =====")
print(f"Epochs parsed: {len(epochs)}")
print(f"Best Val Acc (from log): {best_acc}")
print(f"Final Train Loss: {train_loss[-1]:.4f}")
print(f"Final Val Acc: {val_acc[-1]:.2f}%")

# ========= 画图（服务器保存） =========
# 确保results目录存在
os.makedirs("results", exist_ok=True)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, label="Train Acc")
plt.plot(epochs, val_acc, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Accuracy Curve")

plt.tight_layout()
plt.savefig("results/uatr_epoch_metrics_dis_0204_0.png", dpi=200)
print("Figure saved to uatr_epoch_metrics_dis_0204_0.png")
