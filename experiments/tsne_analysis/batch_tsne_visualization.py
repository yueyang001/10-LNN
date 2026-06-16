#!/usr/bin/env python
"""Compare four student models with t-SNE on the same validation samples.

The script is intended for paper-style comparison between:
1. student baseline
2. the proposed distillation method
3. a logits-based distillation method
4. a feature-based distillation method

All checkpoints are loaded into the same AudioCfC student architecture so the
comparison focuses on the learned student representation.
"""

import argparse
import csv
import inspect
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Add the project root to Python path for proper imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 返回上级目录
sys.path.insert(0, project_root) # 添加项目根目录到 Python 路径

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

@dataclass
class MethodSpec:
    name: str
    checkpoint: str
    checkpoint_key: Optional[str] = None
    p_encoder: Optional[float] = None
    p_classifier: Optional[float] = None


FEATURE_CHOICES = ("logits", "seq_features", "encoder", "pooled_seq")
DEFAULT_CLASS_NAMES = {
    "shipsear": ["Vessel A", "Vessel D", "Vessel C", "Vessel B", "Background Noise"],
    "deepship": ["Passenger Ship", "Tanker", "Cargo Ship", "Tug"],
}


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str, config_dir: str) -> str:
    if os.path.isabs(path):
        return path
    cwd_path = os.path.normpath(path)
    if os.path.exists(cwd_path):
        return cwd_path
    config_path = os.path.normpath(os.path.join(config_dir, path))
    if os.path.exists(config_path):
        return config_path
    return cwd_path


def build_methods(config: dict, config_dir: str) -> List[MethodSpec]:
    methods = []
    for item in config.get("methods", []):
        methods.append(
            MethodSpec(
                name=item["name"],
                checkpoint=resolve_path(item["checkpoint"], config_dir),
                checkpoint_key=item.get("checkpoint_key"),
                p_encoder=item.get("p_encoder"),
                p_classifier=item.get("p_classifier"),
            )
        )
    if len(methods) != 4:
        raise ValueError(f"Expected exactly 4 methods, got {len(methods)}")
    return methods


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def pick_state_dict(ckpt: object, checkpoint_key: Optional[str]) -> Dict[str, torch.Tensor]:
    if checkpoint_key:
        if not isinstance(ckpt, dict) or checkpoint_key not in ckpt:
            raise KeyError(f"Checkpoint does not contain key: {checkpoint_key}")
        return ckpt[checkpoint_key]

    if isinstance(ckpt, dict):
        for key in ("student_state_dict", "model_state_dict", "state_dict"):
            if key in ckpt:
                return ckpt[key]
        if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt

    raise ValueError(
        "Cannot find a model state dict. Set checkpoint_key in the YAML if this "
        "checkpoint uses a custom key."
    )


def build_model(config: dict, method: MethodSpec, device: torch.device) -> torch.nn.Module:
    from models.LNN import AudioCfC

    model_cfg = config["model"]
    student_cfg = model_cfg.get("student", {})
    p_encoder = method.p_encoder
    if p_encoder is None:
        p_encoder = student_cfg.get("p_encoder", model_cfg.get("p_encoder", 0.2))
    p_classifier = method.p_classifier
    if p_classifier is None:
        p_classifier = student_cfg.get("p_classifier", model_cfg.get("p_classifier", 0.3))

    model = AudioCfC(
        num_classes=model_cfg["num_classes"],
        p_encoder=float(p_encoder),
        p_classifier=float(p_classifier),
    ).to(device)

    ckpt = torch.load(method.checkpoint, map_location=device)
    state_dict = strip_module_prefix(pick_state_dict(ckpt, method.checkpoint_key))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] {method.name}: missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] {method.name}: unexpected keys: {len(unexpected)}")

    model.eval()
    return model


def get_student_input(input_data, device: torch.device) -> torch.Tensor:
    if isinstance(input_data, (list, tuple)):
        return input_data[0].to(device)
    return input_data.to(device)


def flatten_feature(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "pooled_seq" and x.dim() == 3:
        return x.mean(dim=1)
    if x.dim() > 2:
        return x.reshape(x.shape[0], -1)
    return x


def select_feature(outputs: Tuple[torch.Tensor, ...], feature_layer: str) -> torch.Tensor:
    logits, seq_features, _, _, _, _, encoder = outputs
    if feature_layer == "logits":
        return logits
    if feature_layer == "seq_features":
        return flatten_feature(seq_features, "seq_features")
    if feature_layer == "encoder":
        return flatten_feature(encoder, "encoder")
    if feature_layer == "pooled_seq":
        return flatten_feature(seq_features, "pooled_seq")
    raise ValueError(f"Unknown feature layer: {feature_layer}")


def make_loader(config: dict, max_samples: int, seed: int, batch_size: int) -> DataLoader:
    from datasets.audio_dataset import AudioDataset, validation_test_transform

    dataset_cfg = config["dataset"]
    dataset = AudioDataset(
        data_dir=dataset_cfg["data_dir"],
        data_flag=dataset_cfg.get("split", "validation"),
        data_type=dataset_cfg.get("data_type", "wav_s"),
        transform=validation_test_transform,
    )

    sample_count = min(max_samples, len(dataset)) if max_samples else len(dataset)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))[:sample_count].tolist()
    subset = Subset(dataset, indices)

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(dataset_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )


def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    feature_layer: str,
) -> Tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    with torch.no_grad():
        for input_data, batch_labels in loader:
            student_input = get_student_input(input_data, device)
            outputs = model(student_input)
            batch_features = select_feature(outputs, feature_layer)
            features.append(batch_features.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def apply_tsne(features: np.ndarray, perplexity: int, seed: int, n_iter: int) -> np.ndarray:
    if features.shape[0] < 4:
        raise ValueError("t-SNE needs at least 4 samples.")

    scaled = StandardScaler().fit_transform(features)
    effective_perplexity = max(2, min(perplexity, (features.shape[0] - 1) // 3))

    kwargs = {
        "n_components": 2,
        "random_state": seed,
        "perplexity": effective_perplexity,
        "learning_rate": "auto",
        "init": "pca",
    }
    tsne_signature = inspect.signature(TSNE)
    if "max_iter" in tsne_signature.parameters:
        kwargs["max_iter"] = n_iter
    else:
        kwargs["n_iter"] = n_iter

    return TSNE(**kwargs).fit_transform(scaled)


def class_colors(num_classes: int):
    if num_classes <= 10:
        return plt.cm.tab10(np.linspace(0, 1, num_classes))
    if num_classes <= 20:
        return plt.cm.tab20(np.linspace(0, 1, num_classes))
    return plt.cm.hsv(np.linspace(0, 1, num_classes))


def infer_dataset_key(config: dict) -> Optional[str]:
    dataset_cfg = config.get("dataset", {})
    text = " ".join(
        str(value)
        for value in (
            config.get("dataset_name"),
            dataset_cfg.get("name"),
            dataset_cfg.get("data_dir"),
            config.get("output_dir"),
        )
        if value
    ).lower()
    if "shipsear" in text or "shipear" in text:
        return "shipsear"
    if "deepship" in text:
        return "deepship"
    return None


def get_class_names(config: dict, num_classes: int) -> List[str]:
    class_names = config.get("class_names")
    if class_names is None:
        class_names = config.get("plot", {}).get("class_names")
    if class_names is None:
        dataset_key = infer_dataset_key(config)
        if dataset_key and len(DEFAULT_CLASS_NAMES[dataset_key]) == num_classes:
            return DEFAULT_CLASS_NAMES[dataset_key]
        return [f"Class {idx}" for idx in range(num_classes)]
    if len(class_names) != num_classes:
        raise ValueError(f"class_names must contain {num_classes} names, got {len(class_names)}")
    return [str(name) for name in class_names]


def legend_columns(class_names: List[str], max_columns: int) -> int:
    if any(len(name) > 28 for name in class_names):
        return min(max_columns, 2)
    return min(max_columns, len(class_names))


def config_bool(config: dict, key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def config_figsize(plot_cfg: dict, default: Tuple[float, float], key: str = "comparison_figsize") -> Tuple[float, float]:
    value = plot_cfg.get(key, default)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    else:
        parts = value
    if len(parts) != 2:
        raise ValueError("comparison_figsize must contain width and height.")
    return float(parts[0]), float(parts[1])


def plot_four_methods(
    tsne_by_method: Dict[str, np.ndarray],
    labels: np.ndarray,
    num_classes: int,
    feature_layer: str,
    output_path: str,
    class_names: List[str],
    legend_fontsize: int = 16,
    xlabel: str = "t-SNE 1",
    ylabel: str = "t-SNE 2",
    show_legend: bool = True,
    plot_cfg: Optional[dict] = None,
) -> None:
    plot_cfg = plot_cfg or {}
    compact_layout = config_bool(plot_cfg, "compact_layout", False)
    fig, axes = plt.subplots(2, 2, figsize=config_figsize(plot_cfg, (13, 12.5)))
    axes = axes.flatten()
    colors = class_colors(num_classes)
    title_fontsize = float(plot_cfg.get("title_fontsize", 13))
    point_size = float(plot_cfg.get("point_size", 18))
    point_alpha = float(plot_cfg.get("point_alpha", 0.72))
    edge_linewidth = float(plot_cfg.get("edge_linewidth", 0.25))
    show_grid = config_bool(plot_cfg, "show_grid", True)
    hide_ticks = config_bool(plot_cfg, "hide_ticks", False)
    show_axis_labels = config_bool(plot_cfg, "show_axis_labels", True)

    for ax, (method_name, points) in zip(axes, tsne_by_method.items()):
        for cls in range(num_classes):
            mask = labels == cls
            if np.any(mask):
                ax.scatter(
                    points[mask, 0],
                    points[mask, 1],
                    c=[colors[cls]],
                    label=class_names[cls] if show_legend else None,
                    s=point_size,
                    alpha=point_alpha,
                    linewidths=edge_linewidth,
                    edgecolors="black",
                )
        ax.set_title(method_name, fontsize=title_fontsize, fontweight="normal" if compact_layout else "bold", pad=2)
        if show_axis_labels:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        if hide_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(length=0)
        ax.grid(show_grid, linestyle="--", alpha=0.25)
        if compact_layout:
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

    bottom = 0.08
    if show_legend:
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=legend_columns(class_names, 3),
            fontsize=legend_fontsize,
            markerscale=1.8,
            borderpad=0.6,
            labelspacing=0.6,
            handletextpad=0.5,
        )
        bottom = 0.18
    # fig.suptitle(f"Four-Method t-SNE Comparison ({feature_layer})", fontsize=16, fontweight="bold")
    if compact_layout:
        fig.subplots_adjust(
            left=float(plot_cfg.get("left", 0.02)),
            right=float(plot_cfg.get("right", 0.99)),
            bottom=float(plot_cfg.get("bottom", 0.02)),
            top=float(plot_cfg.get("top", 0.94)),
            wspace=float(plot_cfg.get("wspace", 0.03)),
            hspace=float(plot_cfg.get("hspace", 0.18)),
        )
    else:
        plt.tight_layout(rect=[0, bottom, 1, 0.96])
    plt.savefig(output_path, dpi=int(plot_cfg.get("dpi", 300)), bbox_inches="tight", pad_inches=float(plot_cfg.get("pad_inches", 0.02)))
    plt.close()


def plot_single_method(
    method_name: str,
    points: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    output_path: str,
    class_names: List[str],
    legend_fontsize: int = 15,
    xlabel: str = "t-SNE 1",
    ylabel: str = "t-SNE 2",
    show_legend: bool = False,
    plot_cfg: Optional[dict] = None,
) -> None:
    plot_cfg = plot_cfg or {}
    compact_layout = config_bool(plot_cfg, "compact_layout", False)
    fig, ax = plt.subplots(figsize=config_figsize(plot_cfg, (2.0, 1.75), "single_figsize"))
    colors = class_colors(num_classes)
    title_fontsize = float(plot_cfg.get("title_fontsize", 9))
    point_size = float(plot_cfg.get("point_size", 3))
    point_alpha = float(plot_cfg.get("point_alpha", 0.85))
    edge_linewidth = float(plot_cfg.get("edge_linewidth", 0))
    show_grid = config_bool(plot_cfg, "show_grid", False)
    hide_ticks = config_bool(plot_cfg, "hide_ticks", True)
    show_axis_labels = config_bool(plot_cfg, "show_axis_labels", False)
    for cls in range(num_classes):
        mask = labels == cls
        if np.any(mask):
            ax.scatter(
                points[mask, 0],
                points[mask, 1],
                c=[colors[cls]],
                label=class_names[cls] if show_legend else None,
                s=point_size,
                alpha=point_alpha,
                linewidths=edge_linewidth,
                edgecolors="black",
            )
    ax.set_title(method_name, fontsize=title_fontsize, fontweight="normal" if compact_layout else "bold", pad=2)
    if show_axis_labels:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(length=0)
    if show_legend:
        ax.legend(
            fontsize=legend_fontsize,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=1,
            markerscale=1.6,
            borderpad=0.6,
            labelspacing=0.6,
        )
    ax.grid(show_grid, linestyle="--", alpha=0.25)
    if compact_layout:
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        fig.subplots_adjust(
            left=float(plot_cfg.get("single_left", 0.02)),
            right=float(plot_cfg.get("single_right", 0.99)),
            bottom=float(plot_cfg.get("single_bottom", 0.02)),
            top=float(plot_cfg.get("single_top", 0.88)),
        )
    else:
        fig.tight_layout()
    fig.savefig(output_path, dpi=int(plot_cfg.get("dpi", 600)), bbox_inches="tight", pad_inches=float(plot_cfg.get("pad_inches", 0.01)))
    plt.close(fig)


def safe_name(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")


def get_filename_suffix(config: dict) -> str:
    suffix = config.get("filename_suffix")
    if suffix is None:
        suffix = config.get("plot", {}).get("filename_suffix")
    if suffix is None or str(suffix).strip() == "":
        return ""
    return f"_{safe_name(str(suffix))}"


def save_manifest(
    output_dir: str,
    methods: List[MethodSpec],
    feature_layer: str,
    labels: np.ndarray,
    feature_shapes: Dict[str, Tuple[int, ...]],
    filename_suffix: str = "",
) -> None:
    manifest_name = f"tsne_run_summary{filename_suffix}.csv"
    with open(os.path.join(output_dir, manifest_name), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "checkpoint", "feature_layer", "feature_shape", "num_samples"])
        for method in methods:
            writer.writerow(
                [
                    method.name,
                    method.checkpoint,
                    feature_layer,
                    "x".join(str(v) for v in feature_shapes[method.name]),
                    len(labels),
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Four-method t-SNE comparison for AudioCfC students.")
    parser.add_argument("--config", required=True, help="YAML file describing dataset and four checkpoints.")
    parser.add_argument("--output", default=None, help="Output directory. Overrides config output_dir.")
    parser.add_argument("--feature_layer", default=None, choices=FEATURE_CHOICES, help="Feature to visualize.")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples from the split.")
    parser.add_argument("--perplexity", type=int, default=None, help="t-SNE perplexity.")
    parser.add_argument("--batch_size", type=int, default=None, help="Inference batch size.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sample selection and t-SNE.")
    parser.add_argument("--n_iter", type=int, default=None, help="t-SNE optimization iterations.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    config_dir = os.path.dirname(os.path.abspath(args.config))
    methods = build_methods(config, config_dir)

    output_dir = args.output or config.get("output_dir", "results/tsne_four_methods")
    feature_layer = args.feature_layer or config.get("feature_layer", "logits")
    max_samples = args.max_samples if args.max_samples is not None else int(config.get("max_samples", 2000))
    perplexity = args.perplexity if args.perplexity is not None else int(config.get("perplexity", 30))
    batch_size = args.batch_size if args.batch_size is not None else int(config.get("batch_size", 32))
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    n_iter = args.n_iter if args.n_iter is not None else int(config.get("n_iter", 1000))

    setup_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Feature layer: {feature_layer}")
    print(f"Output: {output_dir}")

    loader = make_loader(config, max_samples, seed, batch_size)
    num_classes = int(config["model"]["num_classes"])
    class_names = get_class_names(config, num_classes)
    plot_cfg = config.get("plot", {})
    legend_fontsize = int(plot_cfg.get("legend_fontsize", 16))
    xlabel = str(plot_cfg.get("xlabel", "t-SNE 1"))
    ylabel = str(plot_cfg.get("ylabel", "t-SNE 2"))
    show_legend = config_bool(plot_cfg, "show_legend", False)
    save_comparison = config_bool(plot_cfg, "save_comparison", False)
    filename_suffix = get_filename_suffix(config)

    labels_ref = None
    tsne_by_method: Dict[str, np.ndarray] = {}
    feature_shapes: Dict[str, Tuple[int, ...]] = {}

    for method in methods:
        if not os.path.exists(method.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found for {method.name}: {method.checkpoint}")

        print(f"\nLoading method: {method.name}")
        print(f"Checkpoint: {method.checkpoint}")
        model = build_model(config, method, device)
        features, labels = extract_features(model, loader, device, feature_layer)
        feature_shapes[method.name] = features.shape

        if labels_ref is None:
            labels_ref = labels
        elif not np.array_equal(labels_ref, labels):
            raise RuntimeError("Label order changed between models; comparison would be invalid.")

        print(f"Features: {features.shape}. Computing t-SNE...")
        tsne_by_method[method.name] = apply_tsne(features, perplexity, seed, n_iter)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    labels_ref = labels_ref if labels_ref is not None else np.array([])
    comparison_path = os.path.join(output_dir, f"tsne_four_methods_{feature_layer}{filename_suffix}.png")
    if save_comparison:
        plot_four_methods(
            tsne_by_method,
            labels_ref,
            num_classes,
            feature_layer,
            comparison_path,
            class_names,
            legend_fontsize,
            xlabel,
            ylabel,
            show_legend,
            plot_cfg,
        )

    for method_name, points in tsne_by_method.items():
        plot_single_method(
            method_name,
            points,
            labels_ref,
            num_classes,
            os.path.join(output_dir, f"tsne_{safe_name(method_name)}_{feature_layer}{filename_suffix}.png"),
            class_names,
            max(legend_fontsize - 1, 10),
            xlabel,
            ylabel,
            show_legend,
            plot_cfg,
        )

    np.savez(
        os.path.join(output_dir, f"tsne_four_methods_{feature_layer}{filename_suffix}.npz"),
        labels=labels_ref,
        **{safe_name(name): points for name, points in tsne_by_method.items()},
    )
    save_manifest(output_dir, methods, feature_layer, labels_ref, feature_shapes, filename_suffix)

    print("\nDone.")
    if save_comparison:
        print(f"Comparison figure: {comparison_path}")
    print(f"Summary CSV: {os.path.join(output_dir, f'tsne_run_summary{filename_suffix}.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
