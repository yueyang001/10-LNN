"""绘制绿色框中模型的 FLOPs–ACC 对比图。"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator


# 保留 SVG 中的可编辑文字，并确保 PDF 使用 TrueType 字体。
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update(
    {
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "flops_acc_data.csv"
OUTPUT_BASE = ROOT / "flops_acc_comparison"

COLORS = {
    "UATC-DenseNet": "#9AA0A6",
    "1DCT": "#6F7D8C",
    "MSRDN": "#5B7FCA",
    "LNN (w/ KD)": "#C2185B",
}

LABEL_OFFSETS = {
    "DeepShip": {
        "UATC-DenseNet": (5, 5),
        "1DCT": (5, -10),
        "MSRDN": (-6, 6),
        "LNN (w/ KD)": (6, 6),
    },
    "ShipsEar": {
        "UATC-DenseNet": (5, -10),
        "1DCT": (5, 5),
        "MSRDN": (-6, 6),
        "LNN (w/ KD)": (6, -11),
    },
}


def load_data(path: Path):
    """读取源数据并转换数值字段。"""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["flops_mflops"] = float(row["flops_mflops"])
        row["deepship_acc"] = float(row["deepship_acc"])
        row["shipsear_acc"] = float(row["shipsear_acc"])
    return rows


def format_flops(value, _position):
    """以紧凑形式显示对数轴刻度。"""
    return f"{value:,.0f}" if value >= 1000 else f"{value:.0f}"


def draw_panel(ax, rows, dataset, metric_key, panel_label):
    """绘制单个数据集的 FLOPs–ACC 散点面板。"""
    for row in rows:
        method = row["method"]
        is_lnn = method == "LNN (w/ KD)"
        marker = "*" if is_lnn else "o"
        size = 95 if is_lnn else 37
        edge = "#7A1040" if is_lnn else "white"

        ax.scatter(
            row["flops_mflops"],
            row[metric_key],
            s=size,
            marker=marker,
            color=COLORS[method],
            edgecolor=edge,
            linewidth=0.8,
            zorder=3,
        )
        dx, dy = LABEL_OFFSETS[dataset][method]
        ax.annotate(
            method,
            (row["flops_mflops"], row[metric_key]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            fontsize=6.5,
            color="#8E1648" if is_lnn else "#333333",
            fontweight="bold" if is_lnn else "normal",
        )

    ax.set_xscale("log")
    ax.set_xlim(40, 6500)
    ax.set_ylim(79.5, 94.5)
    ax.set_title(dataset, pad=5, fontweight="bold")
    ax.set_xlabel("FLOPs (MFLOPs, log scale)")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(FuncFormatter(format_flops))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0)))
    ax.yaxis.set_major_locator(MultipleLocator(2.5))
    ax.grid(axis="both", which="major", color="#D9D9D9", linewidth=0.6, alpha=0.75)
    ax.grid(axis="x", which="minor", color="#EEEEEE", linewidth=0.4, alpha=0.65)
    ax.set_axisbelow(True)
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


def main():
    rows = load_data(DATA_PATH)
    fig, axes = plt.subplots(1, 2, figsize=(7.20, 3.05), sharey=True)

    draw_panel(axes[0], rows, "DeepShip", "deepship_acc", "a")
    draw_panel(axes[1], rows, "ShipsEar", "shipsear_acc", "b")
    axes[0].set_ylabel("Accuracy (%)")

    # 强调 LNN 相比高计算量模型的效率优势。
    axes[0].annotate(
        "Lowest FLOPs\nand highest ACC",
        xy=(58.97, 93.41),
        xytext=(130, 90.8),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": "#C2185B", "lw": 0.9},
        fontsize=6.5,
        color="#8E1648",
        ha="left",
        va="top",
    )
    axes[1].annotate(
        "Near-best ACC\nat 1/73 FLOPs of MSRDN",
        xy=(58.97, 92.39),
        xytext=(145, 89.7),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": "#C2185B", "lw": 0.9},
        fontsize=6.5,
        color="#8E1648",
        ha="left",
        va="top",
    )

    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.19, top=0.90, wspace=0.14)
    fig.savefig(OUTPUT_BASE.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUTPUT_BASE.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT_BASE.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_BASE.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
