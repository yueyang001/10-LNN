
# 运行脚本（都使用GPUs 4,5,6,7）
python experiments/ablation/grid_search_A.py  # DeepShip消融
python experiments/ablation/grid_search.py    # ShipEar消融

脚本特点：

仅搜索 distill_type (4种组合)，其他参数固定
支持断点续跑（产生 finished.flag 标志）
自动保存结果到 results/ablation_*/ablation_results.csv
## 从项目根目录运行
```
cd d:\a_Program_Projects\10-LNN\UATR\10-LNN
```
```
python experiments/ablation/grid_search_A.py
```
## 结果会保存到：
```
d:\a_Program_Projects\10-LNN\UATR\10-LNN\results\ablation_deepship\ablation_results.csv
```