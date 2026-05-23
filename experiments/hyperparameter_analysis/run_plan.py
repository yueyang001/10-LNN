import argparse
import csv
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]


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


def run_plan(plan_path, gpus=None, dry_run=False):
    with plan_path.open("r", encoding="utf-8") as f:
        plan = yaml.safe_load(f)

    base_config_path = ROOT_DIR / plan["base_config"]
    with base_config_path.open("r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    level = plan["level"]
    train_script = ROOT_DIR / plan["train_script"]
    fixed = plan.get("fixed", {})
    experiments = plan.get("experiments", [])
    checkpoint_root = ROOT_DIR / plan["output"]["checkpoints"]
    results_csv = ROOT_DIR / plan["output"]["results_csv"]
    selected_gpus = gpus or plan.get("default_gpus")

    if not dry_run:
        ensure_results_csv(results_csv)
        checkpoint_root.mkdir(parents=True, exist_ok=True)

    for exp in experiments:
        exp_id = exp["id"]
        exp_name = f"{exp_id}_{exp['name']}"
        save_dir = checkpoint_root / exp_name
        finish_flag = save_dir / "finished.flag"

        if not dry_run and finish_flag.exists():
            print(f"[SKIP] {exp_name} finished")
            continue

        config = apply_changes(base_config, fixed, exp.get("changes", {}))
        config["save"]["save_dir"] = str(save_dir)

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
    parser.add_argument("--dry-run", action="store_true", help="Only print commands")
    args = parser.parse_args()

    run_plan(args.plan, gpus=args.gpus, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
