import argparse
import csv
import socket
import subprocess
from copy import deepcopy
from pathlib import Path

import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]

DATASET_PRESETS = {
    "deepship": {
        "display_name": "DeepShip",
        "base_config": "configs/train_distillation_deepship.yaml",
        "train_script": "train_distillation_deepship.py",
        "fixed": {
            "training.batch_size": 16,
            "training.lr": 0.0004,
            "training.weight_decay": 0.00008,
            "model.student.p_encoder": 0.1,
            "model.student.p_classifier": 0.35,
        },
    },
    "shipsear": {
        "display_name": "ShipSEAR",
        "base_config": "configs/train_distillation_shipsear.yaml",
        "train_script": "train_distillation_shipsear.py",
        "fixed": {
            "training.batch_size": 20,
            "training.lr": 0.0005,
            "training.weight_decay": 0.0001,
            "model.student.p_encoder": 0.355,
            "model.student.p_classifier": 0.255,
        },
    },
}


def set_by_dot_path(config, dot_path, value):
    """按 a.b.c 写入嵌套配置。"""
    current = config
    parts = dot_path.split(".")
    for part in parts[:-1]:
        if part not in current or current[part] is None:
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def apply_changes(base_config, fixed, changes):
    config = deepcopy(base_config)
    for dot_path, value in fixed.items():
        set_by_dot_path(config, dot_path, value)
    for dot_path, value in changes.items():
        set_by_dot_path(config, dot_path, value)
    return config


def flatten_changes(changes):
    return ";".join(f"{key}={value}" for key, value in sorted(changes.items()))


def normalize_dataset_name(dataset):
    if not dataset:
        return None
    key = dataset.lower().replace("_", "").replace("-", "")
    aliases = {
        "deepship": "deepship",
        "deepship622": "deepship",
        "shipsear": "shipsear",
        "shipsear622": "shipsear",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return aliases[key]


def with_dataset_subdir(path, dataset_key):
    parts = list(path.parts)
    if "hyperparameter_analysis" not in parts:
        return path
    idx = parts.index("hyperparameter_analysis")
    dataset_part = DATASET_PRESETS[dataset_key]["display_name"].lower()
    if len(parts) > idx + 1 and parts[idx + 1] == dataset_part:
        return path
    return Path(*parts[: idx + 1], dataset_part, *parts[idx + 1 :])


def is_port_available(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, int(port)))
        except OSError:
            return False
    return True


def stable_port_seed(*parts):
    text = "::".join(str(part) for part in parts)
    total = 0
    for index, char in enumerate(text):
        total = (total + (index + 1) * ord(char)) % 40000
    return 20000 + total


def resolve_master_port(master_port, level, dataset_key, exp_id, exp_index):
    if master_port not in (None, "auto"):
        return str(master_port)

    # 多个 plan 并行启动时，按数据集、实验层级和实验编号错开端口。
    start_port = stable_port_seed(dataset_key, level, exp_id, exp_index)
    for offset in range(1000):
        port = 20000 + ((start_port - 20000 + offset) % 40000)
        if is_port_available(port):
            return str(port)
    raise RuntimeError("No available master_port found in 20000-59999")


def read_best_metrics(best_model_path):
    if not best_model_path.exists():
        return {"best_acc": None, "best_auc": None, "best_f1": None}
    import torch

    ckpt = torch.load(best_model_path, map_location="cpu")
    return {
        "best_acc": ckpt.get("best_acc"),
        "best_auc": ckpt.get("best_auc"),
        "best_f1": ckpt.get("best_f1"),
    }


def ensure_results_csv(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "level",
            "id",
            "name",
            "changes",
            "best_acc",
            "best_auc",
            "best_f1",
            "save_dir",
        ])


def append_result(path, row):
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def run_plan(plan_path, gpus=None, dataset=None, master_port="auto", dry_run=False):
    with plan_path.open("r", encoding="utf-8") as f:
        plan = yaml.safe_load(f)

    plan_dataset_key = normalize_dataset_name(plan.get("dataset")) or "deepship"
    dataset_key = normalize_dataset_name(dataset) or plan_dataset_key
    dataset_preset = DATASET_PRESETS[dataset_key]

    base_config_path = ROOT_DIR / dataset_preset["base_config"]
    with base_config_path.open("r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    level = plan["level"]
    train_script = ROOT_DIR / dataset_preset["train_script"]
    fixed = deepcopy(plan.get("fixed", {}))
    fixed.update(dataset_preset.get("fixed", {}))
    experiments = plan.get("experiments", [])
    checkpoint_root = ROOT_DIR / plan["output"]["checkpoints"]
    results_csv = ROOT_DIR / plan["output"]["results_csv"]
    if dataset_key != plan_dataset_key:
        checkpoint_root = with_dataset_subdir(checkpoint_root, dataset_key)
        results_csv = with_dataset_subdir(results_csv, dataset_key)
    selected_gpus = gpus or plan.get("default_gpus")

    if not dry_run:
        ensure_results_csv(results_csv)
        checkpoint_root.mkdir(parents=True, exist_ok=True)

    for exp_index, exp in enumerate(experiments):
        exp_id = exp["id"]
        exp_name = f"{exp_id}_{exp['name']}"
        save_dir = checkpoint_root / exp_name
        finish_flag = save_dir / "finished.flag"

        if not dry_run and finish_flag.exists():
            print(f"[SKIP] {exp_name} finished")
            continue

        config = apply_changes(base_config, fixed, exp.get("changes", {}))
        config["save"]["save_dir"] = str(save_dir)
        config.setdefault("distributed", {})
        config["distributed"]["master_port"] = resolve_master_port(
            master_port,
            level,
            dataset_key,
            exp_id,
            exp_index,
        )

        temp_config = save_dir / "config.yaml"
        if not dry_run:
            save_dir.mkdir(parents=True, exist_ok=True)
            with temp_config.open("w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, allow_unicode=True)

        cmd = [
            "python",
            str(train_script),
            "--config",
            str(temp_config),
        ]
        if selected_gpus:
            cmd.extend(["--gpus", str(selected_gpus)])

        print(f"[RUN] {exp_name}")
        print(f"[DATASET] {dataset_preset['display_name']}")
        print(f"[MASTER_PORT] {config['distributed']['master_port']}")
        print(" ".join(cmd))
        if dry_run:
            continue

        ret = subprocess.run(cmd, cwd=str(ROOT_DIR))
        if ret.returncode != 0:
            print(f"[STOP] {exp_name} failed, keep directory for resume")
            break

        finish_flag.write_text("done\n", encoding="utf-8")
        metrics = read_best_metrics(save_dir / "best_student.pth")
        append_result(results_csv, [
            level,
            exp_id,
            exp["name"],
            flatten_changes(exp.get("changes", {})),
            metrics["best_acc"],
            metrics["best_auc"],
            metrics["best_f1"],
            str(save_dir),
        ])
        print(f"[DONE] {exp_name}, best_acc={metrics['best_acc']}")


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter analysis ablation plan.")
    parser.add_argument("--plan", required=True, type=Path, help="Path to plan.yaml")
    parser.add_argument("--gpus", default=None, help='GPU ids, for example "4,5,6,7"')
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_PRESETS.keys()),
        default=None,
        help="Override dataset preset from plan.yaml",
    )
    parser.add_argument(
        "--master-port",
        default="auto",
        help='Distributed master port. Use "auto" to avoid conflicts when plans run in parallel.',
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print commands")
    args = parser.parse_args()

    run_plan(
        args.plan,
        gpus=args.gpus,
        dataset=args.dataset,
        master_port=args.master_port,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
