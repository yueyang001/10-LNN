import os
import itertools
import subprocess
import yaml
import csv
import torch

########################################
# 搜索空间
########################################
search_space = {

    # ===== 训练轮数 =====
    "training.num_epochs": [100],

    # ===== weight decay =====
    "training.weight_decay": [1e-4, 5e-4],

    # ===== batch size =====
    "training.batch_size": [16, 32],

    # ===== 学习率 =====
    "training.lr": [0.5e-3, 1e-3],

    # ===== 蒸馏类型 =====
    "distillation.distill_type": [
        "Tser",
        "MemKD",
        "MTSKD_Temp"
    ],

    # ===== 教师是否训练 =====
    "model.teacher.freeze": [True, False],

    # ===== 动态蒸馏权重 =====
    "distillation.USE_DYNAMIC_DISTILL_WEIGHT": [True, False],

    # ===== LNN dropout =====
    "model.student.p_encoder": [0.1, 0.2, 0.3],
    "model.student.p_classifier": [0.2, 0.3, 0.5],
}

########################################
# 基础config
########################################
base_config_path = "configs/train_distillation_shipsear.yaml"
with open(base_config_path, "r") as f:
    base_config = yaml.safe_load(f)

keys = list(search_space.keys())
values = list(search_space.values())
all_combinations = list(itertools.product(*values))

print(f"\n🔥 Total experiments: {len(all_combinations)}")

########################################
# 总实验目录（固定）
########################################
root_save_dir = "checkpoints/grid_shipsear"
os.makedirs(root_save_dir, exist_ok=True)

########################################
# results.csv
########################################
results_save_dir = "results/grid_shipsear"
os.makedirs(results_save_dir, exist_ok=True)
results_csv = os.path.join(results_save_dir, "results.csv")

if not os.path.exists(results_csv):
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch","wd","batch","lr",
            "distill_type","freeze_teacher",
            "dynamic_weight",
            "p_encoder","p_classifier",
            "best_acc"
        ])

########################################
# 遍历实验
########################################
for idx, combo in enumerate(all_combinations):

    print(f"\n===== Exp {idx+1}/{len(all_combinations)} =====")

    # 重新加载base config
    config = yaml.safe_load(open(base_config_path))

    exp_name_parts = []

    ########################################
    # 写入参数（支持三级key）
    ########################################
    for k, v in zip(keys, combo):

        key_parts = k.split(".")
        d = config

        for part in key_parts[:-1]:
            d = d[part]

        d[key_parts[-1]] = v

        exp_name_parts.append(f"{key_parts[-1]}{v}")

    exp_name = "_".join(exp_name_parts)

    save_dir = os.path.join(root_save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    ########################################
    # 断点检测
    ########################################
    finish_flag = os.path.join(save_dir, "finished.flag")
    if os.path.exists(finish_flag):
        print(f"✔ 已完成，跳过: {exp_name}")
        continue

    ########################################
    # 写config
    ########################################
    config["save"]["save_dir"] = save_dir
    temp_config = os.path.join(save_dir, "config.yaml")

    with open(temp_config, "w") as f:
        yaml.dump(config, f)

    ########################################
    # 启动训练
    ########################################
    cmd = f"python train_distillation_shipsear.py --config {temp_config} --gpus 4,5,6,7"
    ret = subprocess.run(cmd, shell=True)

    ########################################
    # 成功才记录
    ########################################
    if ret.returncode == 0:

        with open(finish_flag, "w") as f:
            f.write("done")

        print(f"✔ 完成: {exp_name}")

        ########################################
        # 读取best acc
        ########################################
        best_model_path = os.path.join(save_dir, "best_student.pth")

        best_acc = None
        if os.path.exists(best_model_path):
            ckpt = torch.load(best_model_path, map_location="cpu")
            best_acc = ckpt.get("best_acc", None)

        ########################################
        # 写入results.csv
        ########################################
        with open(results_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                config["training"]["num_epochs"],
                config["training"]["weight_decay"],
                config["training"]["batch_size"],
                config["training"]["lr"],
                config["distillation"]["distill_type"],
                config["model"]["teacher"]["freeze"],
                config["distillation"]["USE_DYNAMIC_DISTILL_WEIGHT"],
                config["model"]["student"]["p_encoder"],
                config["model"]["student"]["p_classifier"],
                best_acc
            ])

        print(f"📊 已记录结果 acc={best_acc}")

    else:
        print(f"✖ 中断: {exp_name}")
        print("下次运行会自动续跑")

print("\nGrid Search全部完成")