"""DeepShip 气泡图 v3：使用对数压缩后的参数与计算效率坐标。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import plot_parameters_flops_v2 as v2


ROOT = Path(__file__).resolve().parent
OUTPUT_BASE = ROOT / "parameters_flops_bubble_v3"

# 保留 v2 原始数据加载函数，避免替换后递归调用。
_V2_LOAD_DATA = v2.load_data

PANEL_ORDER = ("efficiency", "param-acc", "flops-acc")
PANEL_CONFIGS = {
    "efficiency": {
        "title": "Log-efficiency plane",
        "x_key": "parameter_log_efficiency",
        "y_key": "flops_log_efficiency",
        "size_key": "accuracy",
        "x_label": r"Log parameter efficiency ($1/\log_{10}(1+\mathrm{M})$)",
        "y_label": r"Log compute efficiency ($1/\log_{10}(1+\mathrm{MFLOPs})$)",
        "size_label": "Accuracy (%)",
        "size_legend_title": "Bubble area: Accuracy",
        "suffix": "efficiency",
    },
    "param-acc": {
        "title": "Log parameter efficiency vs accuracy",
        "x_key": "parameter_log_efficiency",
        "y_key": "accuracy",
        "size_key": "flops_efficiency",
        "x_label": r"Log parameter efficiency ($1/\log_{10}(1+\mathrm{M})$)",
        "y_label": "DeepShip accuracy (%)",
        "size_label": "Compute efficiency (1/MFLOPs)",
        "size_legend_title": "Bubble area: 1/FLOPs",
        "suffix": "param_acc",
    },
    "flops-acc": {
        "title": "Log compute efficiency vs accuracy",
        "x_key": "flops_log_efficiency",
        "y_key": "accuracy",
        "size_key": "parameter_efficiency",
        "x_label": r"Log compute efficiency ($1/\log_{10}(1+\mathrm{MFLOPs})$)",
        "y_label": "DeepShip accuracy (%)",
        "size_label": "Parameter efficiency (1/M)",
        "size_legend_title": "Bubble area: 1/Parameters",
        "suffix": "flops_acc",
    },
}


def _log_efficiency(value: float, metric_name: str) -> float:
    """计算 1/log10(1+x)，并拒绝无效指标。"""
    if value <= 0:
        raise ValueError(f"{metric_name} must be positive, got {value!r}")
    denominator = math.log10(1.0 + value)
    if denominator <= 0:
        raise ValueError(f"Invalid logarithmic denominator for {metric_name}: {value!r}")
    return 1.0 / denominator


def validate_v3_rows(rows: list[dict[str, object]]) -> None:
    """确认 LNN 在三个面板的横轴、纵轴和气泡面积指标上均为最大值。"""
    lnn = next((row for row in rows if row["method"] == v2.LNN_METHOD), None)
    if lnn is None:
        raise ValueError("DeepShip data must contain LNN")

    for mode in PANEL_ORDER:
        config = PANEL_CONFIGS[mode]
        for role in ("x", "y", "size"):
            key = config[f"{role}_key"]
            maximum = max(float(row[key]) for row in rows)
            if not math.isclose(float(lnn[key]), maximum, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"LNN must have the largest {role} metric ({key}) in {mode}")


def load_data_v3(path: Path) -> list[dict[str, object]]:
    rows = _V2_LOAD_DATA(path)
    for row in rows:
        row["parameter_log_efficiency"] = _log_efficiency(
            float(row["parameters_m"]), "Parameters"
        )
        row["flops_log_efficiency"] = _log_efficiency(
            float(row["flops_mflops"]), "FLOPs"
        )
    validate_v3_rows(rows)
    return rows


def select_mode_v3() -> str:
    print("DeepShip bubble plot v3")
    print("1. 1/log10(1+Parameters) x 1/log10(1+FLOPs)")
    print("2. 1/log10(1+Parameters) x ACC")
    print("3. 1/log10(1+FLOPs) x ACC")
    print("4. Generate the 1x3 combined overview")
    print("5. Generate the overview and all three single plots")
    try:
        choice = input("Select [1-5, default 4]: ").strip() or "4"
    except (EOFError, KeyboardInterrupt):
        print("\nNo terminal input available; using combined mode.")
        return "combined"
    return v2.MENU_OPTIONS.get(choice, "combined")


def configure_v3() -> None:
    """复用 v2 绘图框架，仅替换 v3 的坐标定义与输出名称。"""
    v2.OUTPUT_BASE = OUTPUT_BASE
    v2.PANEL_CONFIGS = PANEL_CONFIGS
    v2.PANEL_ORDER = PANEL_ORDER
    v2.load_data = load_data_v3
    v2.select_mode = select_mode_v3


def main(argv: list[str] | None = None) -> None:
    configure_v3()
    args = v2.parse_args(argv)
    mode = args.mode or select_mode_v3()
    v2.run(mode)


if __name__ == "__main__":
    main(sys.argv[1:])
