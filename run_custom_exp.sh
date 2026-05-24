#!/bin/bash

# 1. 定义会话名称
SESSION_NAME="multi_exp"

# 2. 检查会话是否存在，不存在则创建
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? != 0 ]; then
    tmux new-session -d -s $SESSION_NAME -n dummy
fi

# 3. 在这里写下你 8 个实验各自的专属命令
# 你可以随意更换脚本名、参数、甚至是指定不同的显卡 (CUDA_VISIBLE_DEVICES)
COMMANDS=(
    "conda activate UATR && CUDA_VISIBLE_DEVICES=0 python train_alex.py --data dataset1 --lr 0.01"
    "conda activate UATR && CUDA_VISIBLE_DEVICES=1 python train_bert.py --data dataset1 --lr 0.005 --epochs 50"
    "conda activate env2 && CUDA_VISIBLE_DEVICES=2 python run_resnet.py --data dataset1 --batch 64"
    "conda activate env2 && CUDA_VISIBLE_DEVICES=3 python run_resnet.py --data dataset1 --batch 128"
    "CUDA_VISIBLE_DEVICES=0 python custom_mod.py --data dataset2 --dropout 0.3"
    "CUDA_VISIBLE_DEVICES=1 python custom_mod.py --data dataset2 --dropout 0.5"
    "python evaluate_baseline.py --data dataset2 --metric accuracy"
    "python evaluate_baseline.py --data dataset2 --metric f1"
)

# 4. 自动遍历数组，为每条命令创建一个独立窗口
for i in "${!COMMANDS[@]}"; do
    # 窗口命名为 exp_0, exp_1, exp_2 ...
    WINDOW_NAME="exp_${i}"
    
    # 创建新窗口
    tmux new-window -t $SESSION_NAME -n "$WINDOW_NAME"
    
    # 把对应的命令发送到该窗口并回车执行
    tmux send-keys -t "$SESSION_NAME:$WINDOW_NAME" "${COMMANDS[$i]}" C-m
done

# 5. 收尾工作：删掉初始的 dummy 窗口并挂载会话
tmux kill-window -t $SESSION_NAME:dummy
tmux attach-session -t $SESSION_NAME