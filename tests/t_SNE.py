# 方式1：sklearn（最简单，工业级稳定）
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

# 假设你的高维数据（n_samples, n_features）
X = np.random.randn(1000, 128)          # 示例数据
y = np.random.randint(0, 10, 1000)      # 标签（可选，用于着色）

# 核心调用
tsne = TSNE(
    n_components=2,         # 通常取 2 或 3
    perplexity=30.0,        # 最重要超参！一般 5~50
    learning_rate='auto',   # 'auto' 是目前最稳的选择
    n_iter=1000,            # 迭代次数
    init='pca',             # 用 PCA 初始化通常收敛更好
    random_state=42,
    method='exact'          # 或 'barnes_hut'（大数据集更快，默认）
)

X_2d = tsne.fit_transform(X)

# 可视化
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', alpha=0.7, s=15)
plt.colorbar(scatter)
plt.title("t-SNE visualization")
plt.xlabel("t-SNE dimension 1")
plt.ylabel("t-SNE dimension 2")
plt.tight_layout()
plt.show()