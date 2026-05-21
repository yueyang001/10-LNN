# t-SNE 特征可视化使用指南

## 功能说明

通用t-SNE特征可视化脚本 (`visualize_tsne_features.py`) 支持以下功能：

### 支持的特征层
- **logits**: 模型最后一层输出的logits
- **encoder**: 学生网络编码器输出的特征
- **cnn_features**: 教师CNN网络输出的特征  
- **combined**: 学生编码器和教师CNN特征的组合

### 可视化模式
1. **按类别着色**: 不同类别用不同颜色表示，便于观察类别聚类
2. **按样本索引着色**: 使用彩虹色显示样本顺序，便于观察样本分布
3. **密度图**: 显示特征空间中的密度分布（可选）
4. **对比图**: 并列显示多种特征的可视化效果（可选）

### 关键参数
- `--config`: 模型配置文件路径（必需）
- `--checkpoint`: 模型检查点路径
- `--feature_layer`: 选择要可视化的特征层
- `--max_samples`: 最大采样数量（默认2000）
- `--perplexity`: t-SNE perplexity参数（默认30）
- `--batch_size`: 批处理大小（默认32）

## 使用示例

### 1. 可视化 Logits 特征

```bash
python visualize_tsne_features.py \
  --config configs/your_config.yaml \
  --checkpoint checkpoints/your_model.pth \
  --output_dir results/tsne_logits \
  --feature_layer logits \
  --max_samples 2000
```

### 2. 可视化编码器特征

```bash
python visualize_tsne_features.py \
  --config configs/your_config.yaml \
  --checkpoint checkpoints/your_model.pth \
  --output_dir results/tsne_encoder \
  --feature_layer encoder \
  --max_samples 2000 \
  --perplexity 50
```

### 3. 可视化CNN特征

```bash
python visualize_tsne_features.py \
  --config configs/your_config.yaml \
  --checkpoint checkpoints/your_model.pth \
  --output_dir results/tsne_cnn_features \
  --feature_layer cnn_features \
  --max_samples 3000
```

### 4. 可视化组合特征

```bash
python visualize_tsne_features.py \
  --config configs/your_config.yaml \
  --checkpoint checkpoints/your_model.pth \
  --output_dir results/tsne_combined \
  --feature_layer combined \
  --max_samples 2000
```

## 输出文件

脚本会在指定的输出目录生成以下文件：

- `tsne_logits_by_class.png`: 按类别着色的logits特征t-SNE图
- `tsne_logits_by_index.png`: 按样本索引着色的logits特征t-SNE图
- `tsne_encoder_by_class.png`: 按类别着色的编码器特征t-SNE图
- `tsne_encoder_by_index.png`: 按样本索引着色的编码器特征t-SNE图
- 等等...

## 性能建议

1. **特征维度**
   - 对于高维特征（>1000维），建议先使用PCA降维到50-100维再进行t-SNE
   - t-SNE内部已经使用PCA进行初始化

2. **样本数量**
   - 建议使用2000-5000个样本以获得好的可视化效果
   - 样本太多会导致计算缓慢，太少可能无法显示数据结构

3. **Perplexity参数**
   - 默认值30适合大多数情况
   - 对于样本数<100，建议使用较小的perplexity（5-10）
   - 对于样本数>5000，可以尝试较大的perplexity（50-100）

4. **计算资源**
   - t-SNE计算是CPU密集型的
   - 对于大样本集可能需要几分钟时间
   - 建议在GPU环境中运行以加快数据加载

## 解释可视化结果

### 好的信号
- 同一类别的点聚集在一起
- 不同类别之间有明显的分离
- 没有孤立的点

### 可能的问题
- 类别重叠：表示特征不具有区分性
- 分散的点：可能是特征降维不足或样本不足
- 环形分布：t-SNE的固有特性，反映的不是真实的高维结构

## 自定义扩展

脚本提供了以下可定制的函数：

- `extract_features()`: 修改以提取其他层的特征
- `plot_tsne_by_class()`: 自定义颜色和样式
- `apply_tsne()`: 调整t-SNE超参数
- `create_comparison_plot()`: 创建多特征对比

## 故障排除

### 内存不足
- 减少 `--max_samples`
- 减少 `--batch_size`

### t-SNE运行缓慢
- 减少 `--max_samples`
- 增加 `--perplexity` 会增加计算时间

### 类别分布不平衡
- 检查数据集中的类别不平衡
- 可以在提取特征时进行过采样或欠采样

