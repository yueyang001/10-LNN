"""绘制 Parameters–FLOPs 双面板准确率气泡图。"""

from __future__ import annotations

import csv
import logging
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter


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
OUTPUT_BASE = ROOT / "parameters_flops_bubble"

# 颜色只编码模型，形状只编码输入类型。
METHOD_COLORS = {
    "UATC-DenseNet": "#4E79A7",
    "1DCT": "#D17A22",
    "MSRDN": "#59A14F",
    "LNN (w/ KD)": "#8064A2",
    "ALSI": "#D05A6E",
    "MHT-UATR": "#2A9D8F",
    "CAF-JT": "#B07AA1",
    "T-F Token ViT": "#A56A43",
}

INPUT_MARKERS = {
    "Mel": "o",
    "Audio signal": "h",
}

# 标签通过同色引导线连接气泡；两个面板分别避让，防止错认。
LABEL_LAYOUT = {
    "deepship_acc": {
        "UATC-DenseNet": (13, -4),
        "1DCT": (13, 5),
        "MSRDN": (-60, 10),
        "LNN (w/ KD)": (29, 0),
        "ALSI": (-5, 15),
        "MHT-UATR": (24, -8),
        "CAF-JT": (-20, 17),
        "T-F Token ViT": (-10, -45),
    },
    "shipsear_acc": {
        "UATC-DenseNet": (16, -4),
        "1DCT": (16, 5),
        "MSRDN": (-82, 18),
        "LNN (w/ KD)": (34, 0),
        "ALSI": (-6, 18),
        "MHT-UATR": (34, -23),
        "CAF-JT": (-58, -8),
        "T-F Token ViT": (-10, -62),
    },
}


def load_data(path: Path):
    """读取并转换源数据。"""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("parameters_m", "flops_mflops", "deepship_acc", "shipsear_acc"):
            row[key] = float(row[key])
    return rows


def normalize_input_type(value: str) -> str:
    """兼容 CSV 中常见的输入类型写法。"""
    normalized = value.strip().lower()
    if normalized in {"mel", "mel/cqt"}:
        return "Mel"
    if "audio" in normalized:
        return "Audio signal"
    raise ValueError(f"Unsupported input type: {value}")


def bubble_area(acc: float) -> float:
    """将 80%–98% ACC 线性映射为便于辨识的气泡面积。"""
    return 100.0 + (acc - 80.0) / 18.0 * 1000.0


def compact_number(value, _position=None):
    """紧凑显示对数轴刻度。"""
    if value >= 1000:
        return f"{value / 1000:g}k"
    if value >= 1:
        return f"{value:g}"
    return f"{value:.1f}"


def add_method_label(ax, row, metric_key: str):
    """用同色引导线将模型名称与对应气泡明确连接。"""
    method = row["method"]
    color = METHOD_COLORS[method]
    dx, dy = LABEL_LAYOUT[metric_key][method]
    label = f"{method} ({row[metric_key]:.2f})"

    annotation = ax.annotate(
        label,
        (row["parameters_m"], row["flops_mflops"]),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left" if dx >= 0 else "right",
        va="center",
        fontsize=6.0,
        color=color,
        fontweight="semibold",
        arrowprops={
            "arrowstyle": "-",
            "color": color,
            "linewidth": 0.7,
            "shrinkA": 2,
            "shrinkB": 3,
        },
        annotation_clip=False,
        zorder=5,
    )
    # 引导线置于气泡下方，视觉上从气泡边缘自然伸出。
    annotation.arrow_patch.set_zorder(2)


def draw_panel(ax, rows, title, metric_key, panel_label):
    """绘制一个数据集面板。"""
    for row in rows:
        method = row["method"]
        acc = row[metric_key]
        area = bubble_area(acc)
        color = METHOD_COLORS[method]
        input_type = normalize_input_type(row["input_type"])

        ax.scatter(
            row["parameters_m"],
            row["flops_mflops"],
            s=area,
            marker=INPUT_MARKERS[input_type],
            color=color,
            edgecolors="none",
            linewidths=0,
            alpha=0.80,
            zorder=3,
        )
        add_method_label(ax, row, metric_key)

    ax.set_xscale("log")
    ax.set_yscale("log")
    # 为边界附近的大气泡和标签保留空间。
    ax.set_xlim(0.20, 50)
    ax.set_ylim(28, 10000)
    ax.xaxis.set_major_locator(FixedLocator([0.3, 0.5, 1, 3, 10, 30]))
    ax.yaxis.set_major_locator(FixedLocator([50, 100, 300, 1000, 3000]))
    ax.xaxis.set_major_formatter(FuncFormatter(compact_number))
    ax.yaxis.set_major_formatter(FuncFormatter(compact_number))
    ax.tick_params(which="minor", length=0)
    ax.grid(which="major", color="#D9DEE3", linewidth=0.6, alpha=0.85)
    ax.set_axisbelow(True)
    # 参考示例为每个面板保留完整黑色边框。
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(1.0)
    ax.set_title(title, pad=5, fontweight="bold")
    ax.set_xlabel("Parameters (M, log scale)")
    ax.text(
        -0.13,
        1.04,
        panel_label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def add_input_legend(ax):
    """在右侧面板右下角添加输入类型图例。"""
    input_handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=marker,
            markersize=7,
            markerfacecolor="#74808C",
            markeredgecolor="none",
            markeredgewidth=0,
            label=input_type,
        )
        for input_type, marker in INPUT_MARKERS.items()
    ]
    legend = ax.legend(
        handles=input_handles,
        title="Input Type",
        loc="lower right",
        ncol=1,
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#000000",
        borderpad=0.6,
        labelspacing=0.5,
        handletextpad=0.6,
        fontsize=6.5,
        title_fontsize=7,
    )
    legend.get_frame().set_linewidth(0.7)


def main():
    rows = load_data(DATA_PATH)
    missing_colors = {row["method"] for row in rows} - METHOD_COLORS.keys()
    if missing_colors:
        raise ValueError(f"Missing colors for methods: {sorted(missing_colors)}")

    fig, axes = plt.subplots(1, 2, figsize=(7.20, 3.15), sharex=True, sharey=True)

    draw_panel(axes[0], rows, "DeepShip", "deepship_acc", "a")
    draw_panel(axes[1], rows, "ShipsEar", "shipsear_acc", "b")
    axes[0].set_ylabel("FLOPs (MFLOPs, log scale)")
    add_input_legend(axes[0])
    add_input_legend(axes[1])

    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.18, top=0.90, wspace=0.13)

    outputs = [
        (OUTPUT_BASE.with_suffix(".svg"), {"bbox_inches": "tight"}),
        (OUTPUT_BASE.with_suffix(".tiff"), {"dpi": 600, "bbox_inches": "tight"}),
        (OUTPUT_BASE.with_suffix(".png"), {"dpi": 300, "bbox_inches": "tight"}),
    ]

    for path, kwargs in outputs:
        fig.savefig(path, **kwargs)

    try:
        fig.savefig(OUTPUT_BASE.with_suffix(".pdf"), bbox_inches="tight")
    except Exception as exc:
        print(f"Skip PDF export: {exc}")

    plt.close(fig)


if __name__ == "__main__":
    main()
