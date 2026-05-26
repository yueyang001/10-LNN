#!/bin/bash

set -e

# 启动指定数据集的四组学生网络消融实验。
# 示例：
#   DATASET=deepship bash run_student_ablation_exp.sh
#   DATASET=shipsear bash run_student_ablation_exp.sh
#   DATASET=shipsear DRY_RUN=1 bash run_student_ablation_exp.sh

CONDA_ENV="${CONDA_ENV:-UATR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-deepship}"
DRY_RUN="${DRY_RUN:-0}"

case "${DATASET}" in
    deepship|shipsear)
        ;;
    *)
        echo "Unsupported DATASET: ${DATASET}. Use deepship or shipsear."
        exit 1
        ;;
esac

SESSION_NAME="${SESSION_NAME:-student_network_ablation_${DATASET}}"
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

GPUS=(
    "${GPU_BRANCH:-4,5,6,7}"
    "${GPU_TEMPORAL:-4,5,6,7}"
    "${GPU_CAPACITY:-4,5,6,7}"
    "${GPU_DRASP:-4,5,6,7}"
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

create_window_and_run() {
    local window_name="$1"
    local command="$2"
    local window_id

    # 使用稳定的 window_id，避免窗口标题被 shell 自动改名后无法定位。
    window_id="$(tmux new-window -d -P -F '#{window_id}' -t "${SESSION_NAME}" -n "${window_name}")"
    tmux set-window-option -t "${window_id}" automatic-rename off >/dev/null
    tmux set-window-option -t "${window_id}" allow-rename off >/dev/null
    tmux send-keys -t "${window_id}" "${command}" C-m
}

tmux has-session -t "${SESSION_NAME}" 2>/dev/null || tmux new-session -d -s "${SESSION_NAME}" -n dummy

for i in "${!PLANS[@]}"; do
    plan_path="${PLANS[$i]}"
    base_window="${WINDOWS[$i]}"

    command="$(build_command "${plan_path}" "${DATASET}" "${GPUS[$i]}")"
    create_window_and_run "${base_window}" "${command}"
done

if tmux list-windows -t "${SESSION_NAME}" -F "#{window_name}" | grep -qx "dummy"; then
    tmux kill-window -t "${SESSION_NAME}:dummy"
fi

tmux attach-session -t "${SESSION_NAME}"
