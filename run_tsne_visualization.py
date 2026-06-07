#!/usr/bin/env python
"""Small launcher for the four-method t-SNE comparison."""

import argparse
import os
import subprocess
import sys


TEMPLATE = """# Four-method t-SNE configuration.
# Fill the four checkpoint paths before running.

output_dir: results/tsne_four_methods_deepship
feature_layer: logits       # logits | seq_features | encoder | pooled_seq
max_samples: 2000
perplexity: 30
batch_size: 32
seed: 42
n_iter: 1000

plot:
  legend_fontsize: 18

class_names:
  - Passenger Ship
  - Tanker
  - Cargo Ship
  - Tug

dataset:
  data_dir: /path/to/DeepShip_622
  split: validation
  data_type: wav_s
  num_workers: 4

model:
  num_classes: 4
  student:
    p_encoder: 0.355
    p_classifier: 0.255

methods:
  - name: Student Baseline
    checkpoint: checkpoints/DeepShip/LNN_xxx/best_model.pth
    checkpoint_key: model_state_dict

  - name: Proposed Distillation
    checkpoint: checkpoints/DeepShip/proposed_xxx/best_student.pth
    checkpoint_key: student_state_dict

  - name: Logits KD
    checkpoint: checkpoints/DeepShip/comparison_distillation_kd/best_student.pth
    checkpoint_key: student_state_dict

  - name: Feature KD
    checkpoint: checkpoints/DeepShip/comparison_distillation_rkd/best_student.pth
    checkpoint_key: student_state_dict
"""


def write_template(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(TEMPLATE)
    print(f"Template written to: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch four-method t-SNE comparison.")
    parser.add_argument(
        "--config",
        default="configs/tsne_four_methods_deepship.yaml",
        help="Four-method t-SNE YAML config.",
    )
    parser.add_argument("--make-template", action="store_true", help="Write a YAML template and exit.")
    parser.add_argument("--output", default=None, help="Override output directory.")
    parser.add_argument("--feature_layer", default=None, choices=["logits", "seq_features", "encoder", "pooled_seq"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--perplexity", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n_iter", type=int, default=None)
    args = parser.parse_args()

    if args.make_template:
        write_template(args.config)
        return 0

    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        print("Create one with:")
        print(f"  python run_tsne_visualization.py --make-template --config {args.config}")
        return 1

    cmd = [sys.executable, "experiments/tsne_analysis/batch_tsne_visualization.py", "--config", args.config]
    for key in ("output", "feature_layer", "max_samples", "perplexity", "batch_size", "seed", "n_iter"):
        value = getattr(args, key)
        if value is not None:
            cmd.extend([f"--{key}", str(value)])

    print("Running:")
    print(" ".join(cmd))
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
