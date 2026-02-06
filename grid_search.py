import os
import itertools
import subprocess
import yaml
from datetime import datetime

########################################
# 搜索空间
########################################
search_space = {
    "distillation.temperature": [2,4,6],
    "distillation.alpha": [0.3,0.5,0.7],
    "training.lr": [1e-4,5e-5],
}

base_config_path = "configs/train_distillation_shipsear.yaml"
with open(base_config_path, "r") as f:
    base_config = yaml.safe_load(f)

keys = list(search_space.keys())
values = list(search_space.values())
all_combinations = list(itertools.product(*values))

print(f"Total experiments: {len(all_combinations)}")

########################################
# 总实验目录（固定！！不要每次新建）
########################################
root_save_dir = "checkpoints"
os.makedirs(root_save_dir, exist_ok=True)

########################################
# 遍历实验
########################################
for idx, combo in enumerate(all_combinations):

    print(f"\n===== Exp {idx+1}/{len(all_combinations)} =====")

    config = yaml.safe_load(open(base_config_path))

    exp_name = []
    for k, v in zip(keys, combo):
        section, param = k.split(".")
        config[section][param] = v
        exp_name.append(f"{param}{v}")

    exp_name = "_".join(exp_name)
    save_dir = os.path.join(root_save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    ########################################
    # 关键：断点判断
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
    # 训练成功才写完成标记
    ########################################
    if ret.returncode == 0:
        with open(finish_flag, "w") as f:
            f.write("done")
        print(f"✔ 完成: {exp_name}")
    else:
        print(f"✖ 中断: {exp_name}")
        print("下次运行会自动从这里继续")

print("\n🔥 Grid Search全部完成")
