"""绘制 DeepShip 三种效率—准确率气泡图，并支持终端选择输出模式。"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import warnings
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator


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
        "axes.titlesize": 8,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "legend.frameon": False,
    }
)


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "parameters_flops_data.csv"
OUTPUT_BASE = ROOT / "parameters_flops_bubble_v2"

MIN_BUBBLE_AREA = 100.0
MAX_BUBBLE_AREA = 1100.0
LNN_METHOD = "LNN (w/ KD)"

METHOD_COLORS = {
    "UATC-DenseNet": "#4E79A7",
    "1DCT": "#D17A22",
    "MSRDN": "#59A14F",
    LNN_METHOD: "#8064A2",
    "ALSI": "#D05A6E",
    "MHT-UATR": "#2A9D8F",
    "CAF-JT": "#B07AA1",
    "T-F Token ViT": "#A56A43",
}

PANEL_CONFIGS = {
    "efficiency": {
        "title": "Efficiency plane",
        "x_key": "parameter_efficiency",
        "y_key": "flops_efficiency",
        "size_key": "accuracy",
        "x_label": "Parameter efficiency (1/M)",
        "y_label": "Compute efficiency (1/MFLOPs)",
        "size_label": "Accuracy (%)",
        "size_legend_title": "Bubble area: Accuracy",
        "suffix": "efficiency",
    },
    "param-acc": {
        "title": "Parameter efficiency vs accuracy",
        "x_key": "parameter_efficiency",
        "y_key": "accuracy",
        "size_key": "flops_efficiency",
        "x_label": "Parameter efficiency (1/M)",
        "y_label": "DeepShip accuracy (%)",
        "size_label": "Compute efficiency (1/MFLOPs)",
        "size_legend_title": "Bubble area: 1/FLOPs",
        "suffix": "param_acc",
    },
    "flops-acc": {
        "title": "Compute efficiency vs accuracy",
        "x_key": "flops_efficiency",
        "y_key": "accuracy",
        "size_key": "parameter_efficiency",
        "x_label": "Compute efficiency (1/MFLOPs)",
        "y_label": "DeepShip accuracy (%)",
        "size_label": "Parameter efficiency (1/M)",
        "size_legend_title": "Bubble area: 1/Parameters",
        "suffix": "flops_acc",
    },
}

PANEL_ORDER = ("efficiency", "param-acc", "flops-acc")

# 标签固定在坐标框内部，数据点坐标保持不变。
LABEL_POSITIONS = {
    "efficiency": {
        "UATC-DenseNet": (0.68, 0.36, "right"),
        "1DCT": (0.84, 0.10, "right"),
        "MSRDN": (0.08, 0.84, "left"),
        LNN_METHOD: (0.82, 0.56, "right"),
        "ALSI": (0.08, 0.77, "left"),
        "MHT-UATR": (0.08, 0.56, "left"),
        "CAF-JT": (0.08, 0.63, "left"),
        "T-F Token ViT": (0.08, 0.70, "left"),
    },
    "param-acc": {
        "UATC-DenseNet": (0.64, 0.42, "right"),
        "1DCT": (0.84, 0.34, "right"),
        "MSRDN": (0.08, 0.78, "left"),
        LNN_METHOD: (0.72, 0.66, "left"),
        "ALSI": (0.08, 0.50, "left"),
        "MHT-UATR": (0.08, 0.40, "left"),
        "CAF-JT": (0.08, 0.60, "left"),
        "T-F Token ViT": (0.08, 0.70, "left"),
    },
    "flops-acc": {
        "UATC-DenseNet": (0.45, 0.52, "right"),
        "1DCT": (0.38, 0.34, "right"),
        "MSRDN": (0.08, 0.78, "left"),
        LNN_METHOD: (0.70, 0.66, "left"),
        "ALSI": (0.08, 0.50, "left"),
        "MHT-UATR": (0.08, 0.40, "left"),
        "CAF-JT": (0.08, 0.60, "left"),
        "T-F Token ViT": (0.08, 0.70, "left"),
    },
}

MENU_OPTIONS = {
    "1": "efficiency",
    "2": "param-acc",
    "3": "flops-acc",
    "4": "combined",
    "5": "all",
}


def load_data(path: Path):
    """读取 DeepShip 数据并计算两个效率指标。"""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        parameters = float(row["parameters_m"])
        flops = float(row["flops_mflops"])
        accuracy = float(row["deepship_acc"])
        if parameters <= 0 or flops <= 0:
            raise ValueError(
                f"Parameters and FLOPs must be positive: {row['method']}"
            )
        row["parameters_m"] = parameters
        row["flops_mflops"] = flops
        row["accuracy"] = accuracy
        row["parameter_efficiency"] = 1.0 / parameters
        row["flops_efficiency"] = 1.0 / flops

    validate_rows(rows)
    return rows


def validate_rows(rows):
    """检查颜色配置，并确认 LNN 在三个指标上均为最大值。"""
    methods = {row["method"] for row in rows}
    missing_colors = methods - METHOD_COLORS.keys()
    if missing_colors:
        raise ValueError(f"Missing colors for methods: {sorted(missing_colors)}")
    if LNN_METHOD not in methods:
        raise ValueError(f"Missing required method: {LNN_METHOD}")

    lnn = next(row for row in rows if row["method"] == LNN_METHOD)
    for metric in ("parameter_efficiency", "flops_efficiency", "accuracy"):
        if lnn[metric] != max(row[metric] for row in rows):
            raise ValueError(f"LNN is not the maximum method for {metric}")


def normalize_input_type(value: str) -> str:
    """兼容 CSV 中的 Mel 和音频信号写法。"""
    normalized = value.strip().lower()
    if normalized in {"mel", "mel/cqt"}:
        return "Mel input"
    if "audio" in normalized:
        return "Other audio input"
    raise ValueError(f"Unsupported input type: {value}")


def marker_for(row) -> str:
    """LNN 使用圆形，Mel 使用六边形，其他音频使用三角形。"""
    if row["method"] == LNN_METHOD:
        return "o"
    if normalize_input_type(row["input_type"]) == "Mel input":
        return "h"
    return "^"


def bubble_area(value: float, minimum: float, maximum: float) -> float:
    """将第三项指标线性映射到原脚本的气泡面积范围。"""
    if maximum <= minimum:
        return (MIN_BUBBLE_AREA + MAX_BUBBLE_AREA) / 2.0
    ratio = (value - minimum) / (maximum - minimum)
    return MIN_BUBBLE_AREA + ratio * (MAX_BUBBLE_AREA - MIN_BUBBLE_AREA)


def compact_value(metric_key: str, value: float) -> str:
    """为坐标轴和大小图例生成紧凑数值。"""
    if metric_key == "accuracy":
        return f"{value:.1f}"
    if metric_key == "flops_efficiency":
        return f"{value:.2e}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def axis_formatter(metric_key: str):
    """按指标范围格式化线性坐标。"""
    def formatter(value, _position=None):
        if abs(value) < 1e-12:
            return "0"
        if metric_key == "flops_efficiency":
            if value < 0.001:
                return f"{value:.4f}"
            return f"{value:.3f}"
        if metric_key == "parameter_efficiency":
            return f"{value:g}"
        return f"{value:g}"

    return FuncFormatter(formatter)


def add_method_label(ax, row, config_key: str, compact: bool):
    """用同色引导线连接算法标签和对应气泡。"""
    method = row["method"]
    config = PANEL_CONFIGS[config_key]
    text_x, text_y, horizontal_alignment = LABEL_POSITIONS[config_key][method]
    annotation = ax.annotate(
        f"{method} ({row['accuracy']:.2f})",
        (row[config["x_key"]], row[config["y_key"]]),
        xytext=(text_x, text_y),
        textcoords="axes fraction",
        ha=horizontal_alignment,
        va="center",
        fontsize=5.2 if compact else 6.2,
        color=METHOD_COLORS[method],
        fontweight="bold" if method == LNN_METHOD else "semibold",
        arrowprops={
            "arrowstyle": "-",
            "color": METHOD_COLORS[method],
            "linewidth": 0.65,
            "shrinkA": 2,
            "shrinkB": 3,
        },
        annotation_clip=True,
        zorder=6,
    )
    annotation.arrow_patch.set_zorder(2)


def set_axis_scale(ax, metric_key: str, values, axis: str):
    """效率轴刻度从 0 开始，并为边界气泡保留显示空间。"""
    if metric_key == "accuracy":
        limits = (78.0, 96.5)
        locator = FixedLocator([80, 83, 86, 89, 92, 95])
    else:
        maximum = max(values)
        limits = (-maximum * 0.05, maximum * 1.18)
        candidates = MaxNLocator(nbins=5, min_n_ticks=4).tick_values(
            0.0, maximum * 1.08
        )
        locator = FixedLocator(
            [value for value in candidates if 0.0 <= value <= limits[1]]
        )

    if axis == "x":
        ax.set_xlim(*limits)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(axis_formatter(metric_key))
    else:
        ax.set_ylim(*limits)
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(axis_formatter(metric_key))


def add_size_legend(ax, rows, config_key: str, compact: bool):
    """在面板下方添加第三项指标的最小、中位、最大气泡图例。"""
    config = PANEL_CONFIGS[config_key]
    metric_key = config["size_key"]
    values = [row[metric_key] for row in rows]
    minimum, maximum = min(values), max(values)
    reference_values = (minimum, median(values), maximum)
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=bubble_area(value, minimum, maximum) ** 0.5 * 0.62,
            markerfacecolor="#9AA7B4",
            markeredgecolor="none",
            markeredgewidth=0,
            label=compact_value(metric_key, value),
        )
        for value in reference_values
    ]
    legend = ax.legend(
        handles=handles,
        title=config["size_legend_title"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22 if compact else -0.19),
        ncol=3,
        columnspacing=0.5,
        handlelength=2.5,
        handletextpad=0.5,
        fontsize=4.8 if compact else 5.8,
        title_fontsize=5.2 if compact else 6.2,
        frameon=False,
    )
    ax.add_artist(legend)


def draw_panel(ax, rows, config_key: str, compact: bool, panel_label=None):
    """绘制一个 DeepShip 指标组合面板。"""
    config = PANEL_CONFIGS[config_key]
    size_values = [row[config["size_key"]] for row in rows]
    size_min, size_max = min(size_values), max(size_values)

    for row in rows:
        method = row["method"]
        is_lnn = method == LNN_METHOD
        ax.scatter(
            row[config["x_key"]],
            row[config["y_key"]],
            s=bubble_area(row[config["size_key"]], size_min, size_max),
            marker=marker_for(row),
            color=METHOD_COLORS[method],
            edgecolors="none",
            linewidths=0,
            alpha=0.92 if is_lnn else 0.78,
            zorder=4 if is_lnn else 3,
        )
        add_method_label(ax, row, config_key, compact)

    set_axis_scale(
        ax,
        config["x_key"],
        [row[config["x_key"]] for row in rows],
        "x",
    )
    set_axis_scale(
        ax,
        config["y_key"],
        [row[config["y_key"]] for row in rows],
        "y",
    )
    ax.set_xlabel(config["x_label"])
    ax.set_ylabel(config["y_label"])
    ax.set_title(config["title"], pad=5, fontweight="bold")
    ax.grid(which="major", color="#D9DEE3", linewidth=0.6, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(1.0)

    if panel_label:
        ax.text(
            -0.14,
            1.04,
            panel_label,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    add_size_legend(ax, rows, config_key, compact)


def marker_legend_handles():
    """生成模型/输入类型图例。"""
    entries = (
        ("o", "LNN (proposed, audio)"),
        ("h", "Mel input"),
        ("^", "Other audio input"),
    )
    return [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=marker,
            markersize=7,
            markerfacecolor="#74808C",
            markeredgecolor="none",
            markeredgewidth=0,
            label=label,
        )
        for marker, label in entries
    ]


def add_shared_marker_legend(fig, compact: bool):
    """在图底部添加共享的形状编码图例。"""
    legend = fig.legend(
        handles=marker_legend_handles(),
        title="Model / Input Type",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        columnspacing=1.0,
        handletextpad=0.5,
        fontsize=5.7 if compact else 6.5,
        title_fontsize=6.2 if compact else 7.0,
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#000000",
    )
    legend.get_frame().set_linewidth(0.7)


def build_single_figure(rows, config_key: str):
    """创建一个独立指标组合图。"""
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    draw_panel(ax, rows, config_key, compact=False)
    fig.suptitle("DeepShip", y=0.975, fontsize=10, fontweight="bold")
    add_shared_marker_legend(fig, compact=False)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.31, top=0.86)
    return fig


def build_combined_figure(rows):
    """创建包含三种指标组合的 1×3 总览图。"""
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6))
    for index, (ax, config_key) in enumerate(zip(axes, PANEL_ORDER)):
        draw_panel(ax, rows, config_key, compact=True, panel_label="abc"[index])
    fig.suptitle("DeepShip", y=0.975, fontsize=10, fontweight="bold")
    add_shared_marker_legend(fig, compact=True)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.31, top=0.86, wspace=0.25)
    return fig


def export_figure(fig, output_base: Path):
    """导出可编辑矢量图和高分辨率预览。"""
    outputs = (
        (output_base.with_suffix(".svg"), {}),
        (output_base.with_suffix(".pdf"), {}),
        (output_base.with_suffix(".tiff"), {"dpi": 600}),
        (output_base.with_suffix(".png"), {"dpi": 300}),
    )
    written = []
    for path, kwargs in outputs:
        fig.savefig(path, bbox_inches="tight", **kwargs)
        written.append(path)
    plt.close(fig)
    return written


def select_mode() -> str:
    """无参数运行时，从终端菜单选择输出模式。"""
    print("\nDeepShip bubble plot v2")
    print("1. 1/Parameters x 1/FLOPs (bubble: ACC)")
    print("2. 1/Parameters x ACC (bubble: 1/FLOPs)")
    print("3. 1/FLOPs x ACC (bubble: 1/Parameters)")
    print("4. Combined 1x3 overview")
    print("5. Combined overview + all single plots")
    try:
        choice = input("Select output mode [4]: ").strip() or "4"
    except (EOFError, KeyboardInterrupt):
        print("\nNo terminal input detected; use combined mode.")
        return "combined"
    if choice not in MENU_OPTIONS:
        print(f"Unsupported choice '{choice}'; use combined mode.")
        return "combined"
    return MENU_OPTIONS[choice]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Draw DeepShip parameter/FLOPs/accuracy bubble plots."
    )
    parser.add_argument(
        "--mode",
        choices=("efficiency", "param-acc", "flops-acc", "combined", "all"),
        help="Output mode; omit to use the interactive menu.",
    )
    return parser.parse_args(argv)


def run(mode: str):
    """按指定模式绘图并返回已生成文件。"""
    rows = load_data(DATA_PATH)
    written = []

    if mode in PANEL_CONFIGS:
        config = PANEL_CONFIGS[mode]
        output_base = OUTPUT_BASE.with_name(
            f"{OUTPUT_BASE.name}_{config['suffix']}"
        )
        written.extend(export_figure(build_single_figure(rows, mode), output_base))
    elif mode == "combined":
        written.extend(export_figure(build_combined_figure(rows), OUTPUT_BASE))
    elif mode == "all":
        written.extend(export_figure(build_combined_figure(rows), OUTPUT_BASE))
        for config_key in PANEL_ORDER:
            config = PANEL_CONFIGS[config_key]
            output_base = OUTPUT_BASE.with_name(
                f"{OUTPUT_BASE.name}_{config['suffix']}"
            )
            written.extend(
                export_figure(build_single_figure(rows, config_key), output_base)
            )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    for path in written:
        print(f"Saved: {path}")
    return written


def main(argv=None):
    args = parse_args(argv)
    mode = args.mode or select_mode()
    run(mode)


if __name__ == "__main__":
    main(sys.argv[1:])
