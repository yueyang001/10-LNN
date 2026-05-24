#!/bin/bash

set -e

# 按 README 并行启动四组超参数分析实验。
# 示例：
#   bash run_custom_exp.sh
#   DATASET=shipsear bash run_custom_exp.sh
#   DRY_RUN=1 bash run_custom_exp.sh

SESSION_NAME="${SESSION_NAME:-hyperparameter_analysis}"
CONDA_ENV="${CONDA_ENV:-UATR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-}"
DRY_RUN="${DRY_RUN:-0}"

GPU_BETA="${GPU_BETA:-4,5,6,7}"
GPU_MTSKD="${GPU_MTSKD:-4,5,6,7}"
GPU_MEMKD="${GPU_MEMKD:-4,5,6,7}"
GPU_Z_RANDOM="${GPU_Z_RANDOM:-4,5,6,7}"

PLAN_ROOT="experiments/hyperparameter_analysis"

PLANS=(
    "${PLAN_ROOT}/01_beta_analysis/plan.yaml"
    "${PLAN_ROOT}/02_mtskd_weight_analysis/plan.yaml"
    "${PLAN_ROOT}/03_memkd_weight_analysis/plan.yaml"
    "${PLAN_ROOT}/04_z_random_range_analysis/plan.yaml"
)

WINDOWS=(
    "01_beta"
    "02_mtskd_weight"
    "03_memkd_weight"
    "04_z_random_range"
)

GPUS=(
    "${GPU_BETA}"
    "${GPU_MTSKD}"
    "${GPU_MEMKD}"
    "${GPU_Z_RANDOM}"
)

build_command() {
    local plan_path="$1"
    local gpu_ids="$2"
    local cmd="source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate ${CONDA_ENV} && ${PYTHON_BIN} ${PLAN_ROOT}/run_plan.py --plan ${plan_path} --gpus ${gpu_ids}"

    if [ -n "${DATASET}" ]; then
        cmd="${cmd} --dataset ${DATASET}"
    fi

    if [ "${DRY_RUN}" = "1" ]; then
        cmd="${cmd} --dry-run"
    fi

    echo "${cmd}"
}

tmux has-session -t "${SESSION_NAME}" 2>/dev/null || tmux new-session -d -s "${SESSION_NAME}" -n dummy

for i in "${!PLANS[@]}"; do
    window_name="${WINDOWS[$i]}"
    command="$(build_command "${PLANS[$i]}" "${GPUS[$i]}")"

    tmux new-window -t "${SESSION_NAME}" -n "${window_name}"
    tmux send-keys -t "${SESSION_NAME}:${window_name}" "${command}" C-m
done

if tmux list-windows -t "${SESSION_NAME}" -F "#{window_name}" | grep -qx "dummy"; then
    tmux kill-window -t "${SESSION_NAME}:dummy"
fi

tmux attach-session -t "${SESSION_NAME}"
