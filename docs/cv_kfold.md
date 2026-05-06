# 核心脚本
1. **`kfold_cross_validation.py`** (280行)
   - 主程序：生成K-Fold划分
   - 输出：平衡的10个Fold的详细txt文档
   - 使用：
   ```
   python experiments\cv\kfold_cross_validation.py --data-dir /media/hdd1/chuxiaohui/AI4Ocean_UATR/DeepShip_622
    --output-dir results/kfold_splits_deepship
    ```