# 核心脚本
1. **`kfold_cross_validation.py`** (280行)
   - 主程序：生成K-Fold划分
   - 输出：平衡的10个Fold的详细txt文档
   - 使用：
   ```
   python experiments/cv/kfold_cross_validation.py --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/DeepShip_622
    --output-dir results/kfold_splits_deepship
    ```
## 生成对于shipsear数据集的划分
    ```
    python experiments/cv/kfold_cross_validation.py --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622 --output-dir results/kfold_splits_shipsear
    ```
   ```
    # 生成第1组划分（seed=42）
   python experiments/cv/kfold_cross_validation.py --seed 42 --output-dir results/kfold_splits_shipsear_group1

   # 生成第2组划分（seed=43）
   python experiments/cv/kfold_cross_validation.py --seed 43 --output-dir results/kfold_splits_shipsear_group2

   # 生成第3组划分（seed=44）
   python experiments/cv/kfold_cross_validation.py --seed 44 --output-dir results/kfold_splits_shipsear_group3

   ```
## 生成对于deepship数据集的划分
```
   # 生成第1组划分（seed=42）
   python experiments/cv/kfold_cross_validation.py --seed 42 --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/DeepShip_622 --output-dir results/kfold_splits_deepship_group1

   # 生成第2组划分（seed=43）
   python experiments/cv/kfold_cross_validation.py --seed 46 --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/DeepShip_622 --output-dir results/kfold_splits_deepship_group2

   # 生成第3组划分（seed=44）
   python experiments/cv/kfold_cross_validation.py --seed 60 --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/DeepShip_622 --output-dir results/kfold_splits_deepship_group3
```
相同的随机种子会产生完全相同的划分结果,因为代码用的是 StratifiedKFold。

## shipsear在3个不同终端上分别运行
```
CUDA_VISIBLE_DEVICES=4,5,6,7 python kfold_shipsear_integration.py --train-all \
  --gpus 0,1,2,3 \
  --splits-dir results/kfold_splits_shipsear_group2 \
  --results-dir results/kfold_cv_shipsear_group2 \
  --checkpoints-dir checkpoints/cv_shipsear_group2
  ```
## deepship在3个不同终端上分别运行
```
# 终端1：第1组10fold，用GPU 0-1
CUDA_VISIBLE_DEVICES=4,5,6,7 python kfold_deepship_integration.py --all \
  --gpus 0,1,2,3 \
  --results-dir results/kfold_cv_deepship_group1 \
  --splits-dir results/kfold_splits_deepship_group1
# 终端2：第2组10fold，用GPU 2-3
CUDA_VISIBLE_DEVICES=4,5,6,7 python kfold_deepship_integration.py --all \
  --gpus 0,1,2,3 \
  --results-dir results/kfold_cv_deepship_group2 \
  --splits-dir results/kfold_splits_deepship_group2 &

# 终端3：第3组10fold，用GPU全部（或其他配置）
CUDA_VISIBLE_DEVICES=4,5,6,7 python kfold_deepship_integration.py --all --gpus 0,1,2,3 --results-dir results/kfold_cv_deepship_group3 --splits-dir results/kfold_splits_deepship_group &
  ```

  cd /media/hdd1/fubohan/Code/UATR && pkill -f "train_distillation_shipsear.py" && sleep 2 && rm -rf results/kfold_cv_deepship_group3 && source activate UATR && CUDA_VISIBLE_DEVICES=4,5,6,7 python kfold_deepship_integration.py --all --gpus 0,1,2,3 --results-dir results/kfold_cv_deepship_group3 --splits-dir results/kfold_splits_deepship_group3