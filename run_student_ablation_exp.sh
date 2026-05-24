#!/bin/bash

set -e

# 同时启动 DeepShip 和 ShipSEAR 的学生网络消融实验。
# 示例：
#   bash run_student_ablation_exp.sh
#   DRY_RUN=1 bash run_student_ablation_exp.sh

SESSION_NAME="${SESSION_NAME:-student_network_ablation}"
CONDA_ENV="${CONDA_ENV:-UATR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"

PLAN_ROOT="experiments/hyperparameter_analysis"

PLANS=(
    "${PLAN_ROOT}/05_student_cfc_branch_ablation/plan.yaml"
    "${PLAN_ROOT}/06_student_temporal_granularity/plan.yaml"
    "${PLAN_ROOT}/07_student_cfc_capacity/plan.yaml"
    "${PLAN_ROOT}/08_student_drasp_ablation/plan.yaml"
)

WINDOWS=(
    "05_branch"
    "06_temporal"
    "07_capacity"
    "08_drasp"
)

DEEPSHIP_GPUS=(
    "${GPU_DEEPSHIP_BRANCH:-4,5,6,7}"
    "${GPU_DEEPSHIP_TEMPORAL:-4,5,6,7}"
    "${GPU_DEEPSHIP_CAPACITY:-4,5,6,7}"
    "${GPU_DEEPSHIP_DRASP:-4,5,6,7}"
)

SHIPSEAR_GPUS=(
    "${GPU_SHIPSEAR_BRANCH:-4,5,6,7}"
    "${GPU_SHIPSEAR_TEMPORAL:-4,5,6,7}"
    "${GPU_SHIPSEAR_CAPACITY:-4,5,6,7}"
    "${GPU_SHIPSEAR_DRASP:-4,5,6,7}"
)

build_command() {
    local plan_path="$1"
    local dataset="$2"
    local gpu_ids="$3"
    local cmd="source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate ${CONDA_ENV} && ${PYTHON_BIN} ${PLAN_ROOT}/run_plan.py --plan ${plan_path} --dataset ${dataset} --gpus ${gpu_ids}"

    if [ "${DRY_RUN}" = "1" ]; then
        cmd="${cmd} --dry-run"
    fi

    echo "${cmd}"
}

tmux has-session -t "${SESSION_NAME}" 2>/dev/null || tmux new-session -d -s "${SESSION_NAME}" -n dummy

for i in "${!PLANS[@]}"; do
    plan_path="${PLANS[$i]}"
    base_window="${WINDOWS[$i]}"

    deepship_command="$(build_command "${plan_path}" "deepship" "${DEEPSHIP_GPUS[$i]}")"
    tmux new-window -t "${SESSION_NAME}" -n "ds_${base_window}"
    tmux send-keys -t "${SESSION_NAME}:ds_${base_window}" "${deepship_command}" C-m

    shipsear_command="$(build_command "${plan_path}" "shipsear" "${SHIPSEAR_GPUS[$i]}")"
    tmux new-window -t "${SESSION_NAME}" -n "se_${base_window}"
    tmux send-keys -t "${SESSION_NAME}:se_${base_window}" "${shipsear_command}" C-m
done

if tmux list-windows -t "${SESSION_NAME}" -F "#{window_name}" | grep -qx "dummy"; then
    tmux kill-window -t "${SESSION_NAME}:dummy"
fi

tmux attach-session -t "${SESSION_NAME}"
