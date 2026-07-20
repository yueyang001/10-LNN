"""绘制 DeepShip 参数效率—准确率气泡图 v3。"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import warnings
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


logging.getLogger("fontTools").setLevel(logging.ERROR)
logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=ImportWarning)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "legend.frameon": True,
    }
)


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "parameters_flops_data.csv"
OUTPUT_BASE = ROOT / "parameters_flops_bubble_v3"
SIZE_LEGEND_BASE = ROOT / "parameters_flops_bubble_v3_size_legend"

SOURCE_METHOD = "LNN (w/ KD)"
DISPLAY_NAMES = {SOURCE_METHOD: "BiTAL"}

MIN_BUBBLE_AREA = 100.0
MAX_BUBBLE_AREA = 1100.0

METHOD_COLORS = {
    "UATC-DenseNet": "#4E79A7",
    "1DCT": "#D17A22",
    "MSRDN": "#59A14F",
    SOURCE_METHOD: "#D05A6E",
    "ALSI": "#8064A2",
    "MHT-UATR": "#2A9D8F",
    "CAF-JT": "#B07AA1",
    "T-F Token ViT": "#A56A43",
}

# 标签位置使用数据坐标，避免左侧密集模型的文字和引导线互相遮挡。
LABEL_POSITIONS = {
    "MSRDN": (1.55, 90.40, "left"),
    "CAF-JT": (1.50, 83.90, "left"),
    "ALSI": (0.72, 87.20, "center"),
    "MHT-UATR": (0.15, 82.75, "left"),
    "UATC-DenseNet": (4.35, 85.35, "left"),
    "1DCT": (6.65, 84.30, "left"),
    SOURCE_METHOD: (7.45, 90.80, "right"),
}

# 等量点偏移使标签在屏幕上沿右上 45° 方向展开。
LABEL_OFFSETS = {
    "T-F Token ViT": (28, 28, "left"),
}


def display_name(method: str) -> str:
    return DISPLAY_NAMES.get(method, method)


def log_parameter_efficiency(parameters_m: float) -> float:
    """计算 1/log10(1+Parameters)。"""
    if parameters_m <= 0:
        raise ValueError(f"Parameters must be positive, got {parameters_m!r}")
    return 1.0 / math.log10(1.0 + parameters_m)


def normalize_input_type(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"mel", "mel/cqt"}:
        return "Mel input"
    if "audio" in normalized:
        return "Other audio input"
    raise ValueError(f"Unsupported input type: {value}")


def load_data(path: Path) -> list[dict[str, object]]:
    """读取 DeepShip 数据并计算绘图指标。"""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        parameters = float(row["parameters_m"])
        flops = float(row["flops_mflops"])
        accuracy = float(row["deepship_acc"])
        if flops <= 0:
            raise ValueError(f"FLOPs must be positive: {row['method']}")
        row["parameters_m"] = parameters
        row["flops_mflops"] = flops
        row["accuracy"] = accuracy
        row["parameter_log_efficiency"] = log_parameter_efficiency(parameters)
        row["flops_efficiency"] = 1.0 / flops

    validate_rows(rows)
    return rows


def validate_rows(rows: list[dict[str, object]]) -> None:
    methods = {str(row["method"]) for row in rows}
    missing_colors = methods - METHOD_COLORS.keys()
    if missing_colors:
        raise ValueError(f"Missing colors for methods: {sorted(missing_colors)}")

    proposed = next((row for row in rows if row["method"] == SOURCE_METHOD), None)
    if proposed is None:
        raise ValueError(f"Missing required method: {SOURCE_METHOD}")

    for metric in ("parameter_log_efficiency", "accuracy", "flops_efficiency"):
        maximum = max(float(row[metric]) for row in rows)
        if not math.isclose(float(proposed[metric]), maximum, rel_tol=1e-12):
            raise ValueError(f"BiTAL must have the largest {metric}")


def marker_for(row: dict[str, object]) -> str:
    if row["method"] == SOURCE_METHOD:
        return "o"
    if normalize_input_type(str(row["input_type"])) == "Mel input":
        return "h"
    return "^"


def bubble_area(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return (MIN_BUBBLE_AREA + MAX_BUBBLE_AREA) / 2.0
    ratio = (value - minimum) / (maximum - minimum)
    return MIN_BUBBLE_AREA + ratio * (MAX_BUBBLE_AREA - MIN_BUBBLE_AREA)


def add_method_label(ax, row: dict[str, object]) -> None:
    method = str(row["method"])
    if method in LABEL_OFFSETS:
        text_x, text_y, horizontal_alignment = LABEL_OFFSETS[method]
        text_coordinates = "offset points"
    else:
        text_x, text_y, horizontal_alignment = LABEL_POSITIONS[method]
        text_coordinates = "data"
    is_proposed = method == SOURCE_METHOD
    annotation = ax.annotate(
        f"{display_name(method)} ({float(row['accuracy']):.2f})",
        (
            float(row["parameter_log_efficiency"]),
            float(row["accuracy"]),
        ),
        xytext=(text_x, text_y),
        textcoords=text_coordinates,
        ha=horizontal_alignment,
        va="center",
        fontsize=6.7 if is_proposed else 6.1,
        color=METHOD_COLORS[method],
        fontweight="bold" if is_proposed else "semibold",
        arrowprops={
            "arrowstyle": "-",
            "color": METHOD_COLORS[method],
            "linewidth": 0.65,
            "shrinkA": 2,
            "shrinkB": 4,
        },
        annotation_clip=True,
        zorder=6,
    )
    annotation.arrow_patch.set_zorder(2)


def marker_legend_handles() -> list[Line2D]:
    entries = (
        ("o", "BiTAL (Ours)", METHOD_COLORS[SOURCE_METHOD]),
        ("h", "Mel input", "#74808C"),
        ("^", "Other audio", "#74808C"),
    )
    return [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=marker,
            markersize=6.2,
            markerfacecolor=color,
            markeredgecolor="none",
            label=label,
        )
        for marker, label, color in entries
    ]


def add_input_legend(ax) -> None:
    """将模型/输入类型图例放在黑色坐标框右下角。"""
    legend = ax.legend(
        handles=marker_legend_handles(),
        title="Model Input Type",
        loc="upper left",
        bbox_to_anchor=(0.015, 0.975),
        borderaxespad=0,
        ncol=1,
        handletextpad=0.55,
        labelspacing=0.45,
        borderpad=0.55,
        fontsize=5.9,
        title_fontsize=6.3,
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#000000",
    )
    legend.get_frame().set_linewidth(0.7)
    legend.set_zorder(10)


def build_main_figure(rows: list[dict[str, object]]):
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    size_values = [float(row["flops_efficiency"]) for row in rows]
    size_min, size_max = min(size_values), max(size_values)

    for row in rows:
        method = str(row["method"])
        is_proposed = method == SOURCE_METHOD
        ax.scatter(
            float(row["parameter_log_efficiency"]),
            float(row["accuracy"]),
            s=bubble_area(float(row["flops_efficiency"]), size_min, size_max),
            marker=marker_for(row),
            color=METHOD_COLORS[method],
            edgecolors="none",
            linewidths=0,
            alpha=0.92 if is_proposed else 0.80,
            zorder=4 if is_proposed else 3,
        )
        add_method_label(ax, row)

    ax.set_xlim(0.0, 9.0)
    ax.set_ylim(78.0, 95.0)
    ax.set_xticks(range(0, 10, 1))
    ax.set_yticks([80, 83, 86, 89, 92, 95])
    # x 轴 1 个单位与 y 轴 3 个百分点使用相同显示长度。
    ax.set_aspect(1.0 / 3.0, adjustable="box")
    ax.set_xlabel(r"Log parameter efficiency ($1/\log_{10}(1 + \mathrm{M})$)")
    ax.set_ylabel("DeepShip accuracy (%)")
    ax.set_title("DeepShip", pad=6, fontsize=10, fontweight="bold")
    ax.grid(which="major", color="#D9DEE3", linewidth=0.6, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(1.0)

    add_input_legend(ax)
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.17, top=0.86)
    return fig


def build_size_legend(rows: list[dict[str, object]]):
    """单独生成气泡面积说明，供论文排版时按需插入。"""
    values = [float(row["flops_efficiency"]) for row in rows]
    minimum, maximum = min(values), max(values)
    reference_values = (minimum, median(values), maximum)
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=math.sqrt(bubble_area(value, minimum, maximum)) * 0.62,
            markerfacecolor="#9AA7B4",
            markeredgecolor="none",
            label=f"{value:.2e}",
        )
        for value in reference_values
    ]

    fig, ax = plt.subplots(figsize=(3.8, 0.95))
    ax.axis("off")
    fig.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.34),
        ncol=3,
        columnspacing=0.9,
        handlelength=2.5,
        handletextpad=0.55,
        fontsize=6.2,
        frameon=False,
    )
    fig.text(
        0.5,
        0.82,
        "Bubble area: compute efficiency (1/FLOPs)",
        ha="center",
        va="top",
        fontsize=6.8,
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    return fig


def export_figure(fig, output_base: Path) -> list[Path]:
    outputs = (
        (output_base.with_suffix(".svg"), {}),
        (output_base.with_suffix(".pdf"), {}),
        (output_base.with_suffix(".tiff"), {"dpi": 600}),
        (output_base.with_suffix(".png"), {"dpi": 300}),
    )
    written = []
    for path, kwargs in outputs:
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        written.append(path)
    plt.close(fig)
    return written


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Draw the refined DeepShip parameter-efficiency bubble plot."
    )
    parser.add_argument(
        "--mode",
        choices=("figure", "size-legend", "all"),
        default="all",
        help="Output the main figure, the standalone size legend, or both.",
    )
    return parser.parse_args(argv)


def run(mode: str) -> list[Path]:
    rows = load_data(DATA_PATH)
    written = []
    if mode in {"figure", "all"}:
        written.extend(export_figure(build_main_figure(rows), OUTPUT_BASE))
    if mode in {"size-legend", "all"}:
        written.extend(export_figure(build_size_legend(rows), SIZE_LEGEND_BASE))

    for path in written:
        print(f"Saved: {path}")
    return written


def main(argv=None) -> None:
    args = parse_args(argv)
    run(args.mode)


if __name__ == "__main__":
    main(sys.argv[1:])
