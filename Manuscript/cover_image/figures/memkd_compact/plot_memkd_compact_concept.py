"""生成 MemKD 紧凑布局概念图。"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BLUE = "#367CF7"
BLUE_LIGHT = "#A9C8FF"
GREEN = "#2E9E44"
GREEN_LIGHT = "#8FD19B"
ORANGE = "#FF6B00"
INK = "#17212B"
MUTED = "#68737D"
CARD = "#F7F9FC"
LINE = "#DCE3EA"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
    }
)


def rounded_card(ax, x, y, w, h, facecolor=CARD, edgecolor=LINE, radius=0.018):
    """绘制统一的圆角信息卡片。"""
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=0.9,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=INK, lw=1.0, style="-|>", mutation=9, ls="-"):
    """按主画布坐标绘制箭头。"""
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        linestyle=ls,
        color=color,
        transform=ax.transAxes,
        shrinkA=0,
        shrinkB=0,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def draw_mini_axes(ax, x0, y0, w, h):
    arrow(ax, (x0, y0), (x0, y0 + h), lw=0.85, mutation=7)
    arrow(ax, (x0, y0), (x0 + w, y0), lw=0.85, mutation=7)


def draw_series(ax, xs, ys, color, marker_face, lw=1.2):
    ax.plot(xs, ys, color=color, lw=lw, transform=ax.transAxes, zorder=3)
    ax.scatter(
        xs,
        ys,
        s=28,
        facecolor=marker_face,
        edgecolor=color,
        linewidth=0.75,
        transform=ax.transAxes,
        zorder=4,
    )
    for i in range(1, len(xs)):
        frac = 0.58
        p0 = (xs[i - 1] + frac * (xs[i] - xs[i - 1]), ys[i - 1] + frac * (ys[i] - ys[i - 1]))
        p1 = (xs[i - 1] + 0.84 * (xs[i] - xs[i - 1]), ys[i - 1] + 0.84 * (ys[i] - ys[i - 1]))
        arrow(ax, p0, p1, color=color, lw=0.9, mutation=6)


def distance_arrow(ax, x, y0, y1):
    """表示教师与学生状态差异。"""
    arrow(ax, (x, y0), (x, y1), color=ORANGE, lw=1.0, style="<|-|>", mutation=7, ls=(0, (3, 2)))


def draw_short_term(ax, box):
    x, y, w, h = box
    ax.text(x + 0.025, y + h - 0.065, "SHORT-TERM", color=ORANGE, fontsize=8.5, fontweight="bold", transform=ax.transAxes)
    ax.text(x + 0.025, y + h - 0.105, r"Adjacent state change  $(z=1)$", color=MUTED, fontsize=7.2, transform=ax.transAxes)

    px, py = x + 0.045, y + 0.155
    pw, ph = w - 0.075, h - 0.30
    draw_mini_axes(ax, px, py, pw, ph)

    xs = np.linspace(px + 0.018, px + pw - 0.018, 6)
    teacher = np.array([0.57, 0.73, 0.66, 0.52, 0.44, 0.41]) * ph + py
    student = np.array([0.18, 0.30, 0.34, 0.24, 0.23, 0.21]) * ph + py
    draw_series(ax, xs, teacher, BLUE, BLUE_LIGHT)
    draw_series(ax, xs, student, GREEN, GREEN_LIGHT)
    distance_arrow(ax, xs[1], student[1] + 0.012, teacher[1] - 0.012)

    ax.text(xs[1] + 0.012, teacher[1] + 0.035, r"$\Delta h_T^{(1)}$", color=BLUE, fontsize=9.5, transform=ax.transAxes)
    ax.text(xs[1] + 0.006, student[1] - 0.070, r"$\Delta h_S^{(1)}$", color=GREEN, fontsize=9.5, transform=ax.transAxes)
    ax.text(x + w / 2, y + 0.048, r"$\mathcal{L}_S=\|\Delta h_T^{(1)}-\Delta h_S^{(1)}\|_2^2$", ha="center", color=INK, fontsize=9.4, transform=ax.transAxes)


def draw_dice(ax, x, y):
    rounded_card(ax, x, y, 0.030, 0.050, facecolor=ORANGE, edgecolor=ORANGE, radius=0.005)
    for dx, dy in [(0.008, 0.012), (0.022, 0.038), (0.015, 0.025)]:
        ax.scatter([x + dx], [y + dy], s=5, color="white", transform=ax.transAxes, zorder=8)


def draw_long_term(ax, box):
    x, y, w, h = box
    ax.text(x + 0.025, y + h - 0.065, "LONG-TERM", color=ORANGE, fontsize=8.5, fontweight="bold", transform=ax.transAxes)
    draw_dice(ax, x + w - 0.115, y + h - 0.115)
    ax.text(x + w - 0.073, y + h - 0.078, r"sample $z$", color=ORANGE, fontsize=8.5, fontweight="bold", transform=ax.transAxes)
    ax.text(x + w - 0.073, y + h - 0.105, r"$z\sim\mathcal{U}\{2,\ldots,T/2\}$", color=MUTED, fontsize=7.0, transform=ax.transAxes)

    px, py = x + 0.045, y + 0.155
    pw, ph = w - 0.075, h - 0.30
    draw_mini_axes(ax, px, py, pw, ph)

    xs = np.linspace(px + 0.018, px + pw - 0.018, 6)
    teacher = np.array([0.54, 0.58, 0.63, 0.63, 0.68, 0.63]) * ph + py
    student = np.array([0.25, 0.20, 0.22, 0.30, 0.41, 0.47]) * ph + py
    draw_series(ax, xs, teacher, BLUE, BLUE_LIGHT)
    draw_series(ax, xs, student, GREEN, GREEN_LIGHT)
    distance_arrow(ax, xs[2], student[2] + 0.012, teacher[2] - 0.012)

    # 用弧线明确表示跨 z 步的长期偏移。
    arc = FancyArrowPatch(
        (xs[0], student[0] - 0.015),
        (xs[4], student[4] - 0.010),
        connectionstyle="arc3,rad=0.34",
        arrowstyle="-|>",
        mutation_scale=8,
        lw=1.25,
        color=GREEN,
        transform=ax.transAxes,
    )
    ax.add_patch(arc)
    ax.text(xs[3] + 0.010, student[3] - 0.105, r"$\Delta h_S^{(z)}$", color=GREEN, fontsize=9.5, transform=ax.transAxes)

    bracket_y = py + ph + 0.025
    ax.plot([xs[0], xs[-2]], [bracket_y, bracket_y], color=ORANGE, lw=1.0, transform=ax.transAxes)
    ax.plot([xs[0], xs[0]], [bracket_y, bracket_y - 0.018], color=ORANGE, lw=1.0, transform=ax.transAxes)
    ax.plot([xs[-2], xs[-2]], [bracket_y, bracket_y - 0.018], color=ORANGE, lw=1.0, transform=ax.transAxes)
    ax.text((xs[0] + xs[-2]) / 2, bracket_y + 0.014, "$z$ steps", ha="center", color=ORANGE, fontsize=7.8, fontweight="bold", transform=ax.transAxes)

    ax.text(x + w / 2, y + 0.048, r"$\mathcal{L}_L=\|\Delta h_T^{(z)}-\Delta h_S^{(z)}\|_2^2$", ha="center", color=INK, fontsize=9.4, transform=ax.transAxes)


def draw_loss_card(ax, box):
    x, y, w, h = box
    rounded_card(ax, x, y, w, h, facecolor="#FFF6EF", edgecolor="#FFD3B2", radius=0.020)
    ax.text(x + 0.030, y + h - 0.070, "MEMORY ALIGNMENT", color=ORANGE, fontsize=8.5, fontweight="bold", transform=ax.transAxes)
    ax.text(x + 0.030, y + h - 0.118, "Two temporal scales, one objective", color=MUTED, fontsize=7.2, transform=ax.transAxes)

    rounded_card(ax, x + 0.030, y + h - 0.260, w - 0.060, 0.086, facecolor="white", edgecolor="#FFE1CA", radius=0.012)
    ax.text(x + w / 2, y + h - 0.217, r"$\lambda_s\,\mathcal{L}_S$", ha="center", va="center", color=INK, fontsize=12, transform=ax.transAxes)
    rounded_card(ax, x + 0.030, y + h - 0.374, w - 0.060, 0.086, facecolor="white", edgecolor="#FFE1CA", radius=0.012)
    ax.text(x + w / 2, y + h - 0.331, r"$\lambda_l\,\mathcal{L}_L$", ha="center", va="center", color=INK, fontsize=12, transform=ax.transAxes)

    arrow(ax, (x + w / 2, y + h - 0.390), (x + w / 2, y + 0.180), color=ORANGE, lw=1.2, mutation=9)
    rounded_card(ax, x + 0.025, y + 0.055, w - 0.050, 0.108, facecolor=ORANGE, edgecolor=ORANGE, radius=0.016)
    ax.text(x + w / 2, y + 0.109, r"$\mathcal{L}_M=\lambda_s\mathcal{L}_S+\lambda_l\mathcal{L}_L$", ha="center", va="center", color="white", fontsize=11.8, fontweight="bold", transform=ax.transAxes)


def main():
    out_dir = Path(__file__).resolve().parent
    fig, ax = plt.subplots(figsize=(11.2, 4.25), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.035, 0.925, "MemKD", color=ORANGE, fontsize=19, fontweight="bold", transform=ax.transAxes)
    ax.text(0.035, 0.875, "Memory-discrepancy knowledge distillation", color=MUTED, fontsize=8.4, transform=ax.transAxes)

    # 全图只保留一个图例，减少重复信息与视觉跳转。
    ax.scatter([0.365, 0.468], [0.916, 0.916], s=34, color=[BLUE, GREEN], edgecolors="white", linewidths=0.6, transform=ax.transAxes)
    ax.text(0.379, 0.916, "Teacher", va="center", color=INK, fontsize=8.1, fontweight="bold", transform=ax.transAxes)
    ax.text(0.482, 0.916, "Student", va="center", color=INK, fontsize=8.1, fontweight="bold", transform=ax.transAxes)
    ax.plot([0.61, 0.645], [0.916, 0.916], color=ORANGE, lw=1.1, ls=(0, (3, 2)), transform=ax.transAxes)
    ax.text(0.653, 0.916, "state discrepancy", va="center", color=INK, fontsize=8.1, transform=ax.transAxes)

    short_box = (0.035, 0.095, 0.285, 0.715)
    long_box = (0.340, 0.095, 0.335, 0.715)
    loss_box = (0.715, 0.095, 0.250, 0.715)
    rounded_card(ax, *short_box)
    rounded_card(ax, *long_box)
    draw_short_term(ax, short_box)
    draw_long_term(ax, long_box)
    draw_loss_card(ax, loss_box)

    # “+”强调短期与长期是并列损失项，不是先后步骤。
    ax.scatter([0.330], [0.455], s=118, facecolor="white", edgecolor=ORANGE, linewidth=1.2, transform=ax.transAxes, zorder=10)
    ax.text(0.330, 0.455, "+", ha="center", va="center", color=ORANGE, fontsize=10.5, fontweight="bold", transform=ax.transAxes, zorder=11)
    arrow(ax, (0.675, 0.455), (0.710, 0.455), color=ORANGE, lw=1.3, mutation=9)

    fig.subplots_adjust(left=0.006, right=0.994, bottom=0.018, top=0.985)
    for suffix in ("svg", "pdf"):
        fig.savefig(out_dir / f"memkd_compact_concept.{suffix}", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_dir / "memkd_compact_concept.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()
