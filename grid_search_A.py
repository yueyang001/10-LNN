import os
import itertools
import subprocess
import yaml
import csv
import torch

########################################
# 消融实验搜索空间（仅搜索模块组合，其余参数固定为最优值）
########################################
search_space = {
    # ===== 训练轮数（固定最优值）=====
    "training.num_epochs": [200],

    # ===== weight decay（固定最优值）=====
    "training.weight_decay": [8e-5],

    # ===== batch size（固定最优值）=====
    "training.batch_size": [16],

    # ===== 学习率（固定最优值）=====
    "training.lr": [0.0004],

    # ===== 蒸馏类型（仅搜索这个变量，对应消融实验的4种模块组合）=====
    "distillation.distill_type": [
        "Tser",         # 时序特征蒸馏
        "MemKD",        # 记忆知识蒸馏
        "MTSKD_Temp",   # 多时序知识蒸馏 + 温度
        "none",         # 无蒸馏（仅使用硬标签）
    ],

    # ===== 教师是否训练（固定最优值）=====
    "model.teacher.freeze": [True],

    # ===== 动态蒸馏权重（固定最优值）=====
    "distillation.USE_DYNAMIC_DISTILL_WEIGHT": [False],

    # ===== LNN dropout（固定最优值）=====
    "model.student.p_encoder": [0.1],
    "model.student.p_classifier": [0.35],
}

########################################
# 基础config
########################################
base_config_path = "configs/train_distillation_deepship.yaml"
with open(base_config_path, "r") as f:
    base_config = yaml.safe_load(f)

keys = list(search_space.keys())
values = list(search_space.values())
all_combinations = list(itertools.product(*values))

print(f"\n🔥 消融实验总数: {len(all_combinations)} (对应4种模块组合)")

########################################
# 总实验目录（固定）
########################################
root_save_dir = "checkpoints/ablation_deepship"  # 改名为消融实验目录，方便区分
os.makedirs(root_save_dir, exist_ok=True)

########################################
# results.csv（消融实验专用）
########################################
results_save_dir = "results/ablation_deepship"
os.makedirs(results_save_dir, exist_ok=True)
results_csv = os.path.join(results_save_dir, "ablation_results.csv")

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
# 遍历消融实验（仅4组，对应4种模块组合）
########################################
for idx, combo in enumerate(all_combinations):

    print(f"\n===== 消融实验 {idx+1}/{len(all_combinations)} =====")

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
    # 启动训练（注意：如果distill_type为None，需确保代码兼容无蒸馏的情况）
    ########################################
    cmd = f"python train_distillation_deepship.py --config {temp_config} --gpus 4,5,6,7"
    ret = subprocess.run(cmd, shell=True)

    ########################################
    # 成功才记录
    ########################################
    if ret.returncode == 0:
        with open(finish_flag, "w") as f:
            f.write("done")
        print(f"✔ 完成: {exp_name}")

        # 读取best acc
        best_model_path = os.path.join(save_dir, "best_student.pth")
        best_acc = None
        if os.path.exists(best_model_path):
            ckpt = torch.load(best_model_path, map_location="cpu")
            best_acc = ckpt.get("best_acc", None)

        # 写入消融实验结果
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

print(f"\n消融实验全部完成！可直接从ablation_results.csv提取{len(search_space['distillation.distill_type'])}组结果制作消融表格")