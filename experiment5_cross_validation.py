import os
import itertools
import subprocess
import yaml
import csv
import torch
import random  # 引入random库
 
########################################
# 十倍交叉验证配置
########################################
search_space = {
    # ===== 交叉验证折叠数 =====
    "dataset.total_folds": [10],
    
    # ===== 当前折索引（0-9）=====
    "dataset.fold": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
 
    # ===== 固定训练参数 =====
    "training.num_epochs": [200],
    "training.weight_decay": [1e-4],
    "training.batch_size": [20],
    "training.lr": [0.0005],
 
    # ===== 蒸馏类型 =====
    "distillation.distill_type": ["MTSKD"], 
 
    # ===== 其他固定参数 =====
    "model.teacher.freeze": [True],
    "distillation.USE_DYNAMIC_DISTILL_WEIGHT": [False],
    "model.student.p_encoder": [0.355],
    "model.student.p_classifier": [0.255],
}
 
########################################
# 基础config路径
########################################
base_config_path = "configs/train_distillation_shipsear.yaml"
 
# 检查配置文件是否存在
if not os.path.exists(base_config_path):
    print(f"❌ 错误: 配置文件 {base_config_path} 不存在，请检查路径。")
    exit(1)
 
with open(base_config_path, "r") as f:
    base_config = yaml.safe_load(f)
 
keys = list(search_space.keys())
values = list(search_space.values())
all_combinations = list(itertools.product(*values))
 
print(f"\n🔥 ShipEar 十倍交叉验证实验总数: {len(all_combinations)} 组")
 
########################################
# 目录设置
########################################
root_save_dir = "checkpoints/cv_shipsear"
os.makedirs(root_save_dir, exist_ok=True)
 
results_save_dir = "results/cv_shipsear"
os.makedirs(results_save_dir, exist_ok=True)
results_csv = os.path.join(results_save_dir, "cv_results.csv")
 
if not os.path.exists(results_csv):
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold_index", "total_folds",
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
 
    print(f"\n===== ShipEar 十倍交叉验证实验 {idx+1}/{len(all_combinations)} =====")
 
    # 重新加载base config
    config = yaml.safe_load(open(base_config_path))
 
    exp_name_parts = []
 
    ########################################
    # 写入参数
    ########################################
    for k, v in zip(keys, combo):
        key_parts = k.split(".")
        d = config
        for part in key_parts[:-1]:
            d = d[part]
        d[key_parts[-1]] = v
        
        if k == "dataset.fold":
            exp_name_parts.append(f"fold{v}")
        else:
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
    # 核心修复：动态修改端口（必须转为字符串 str）
    ########################################
    random_port = random.randint(20000, 60000)
    # 注意：这里必须使用 str()，因为 os.environ 只接受字符串
    config["distributed"]["master_port"] = str(random_port)
    print(f"🚀 当前实验使用端口: {random_port}")
 
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
    # 记录结果
    ########################################
    if ret.returncode == 0:
        with open(finish_flag, "w") as f:
            f.write("done")
        print(f"✔ 完成: {exp_name}")
 
        best_model_path = os.path.join(save_dir, "best_student.pth")
        best_acc = None
        if os.path.exists(best_model_path):
            ckpt = torch.load(best_model_path, map_location="cpu")
            best_acc = ckpt.get("best_acc", None)
 
        with open(results_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                config["dataset"]["fold"],
                config["dataset"]["total_folds"],
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
        print(f"📊 已记录结果 Fold {config['dataset']['fold']}: acc={best_acc}")
    else:
        print(f"✖ 中断: {exp_name}")
        print("下次运行会自动续跑")
 
print("\n✨ ShipEar 十倍交叉验证实验全部完成！")