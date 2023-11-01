import numpy as np
from sklearn.neighbors import NearestNeighbors


def reliefF(X, y, k):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)

    # Find k nearest neighbors for each sample
    neigh = NearestNeighbors(n_neighbors=k + 1)
    neigh.fit(X)
    distances, indices = neigh.kneighbors(X)

    for i in range(n_samples):
        same_class_neighbors = indices[i, 1:]  # Exclude the sample itself
        diff_class_neighbors = np.delete(indices, same_class_neighbors)

        for j in range(n_features):
            feature_diff_same = np.mean(np.abs(X[same_class_neighbors, j] - X[i, j]))
            feature_diff_diff = np.mean(np.abs(X[diff_class_neighbors, j] - X[i, j]))
            weights[j] += feature_diff_diff - feature_diff_same

    weights /= n_samples

    return weights


# 示例数据
X = np.array([[1, 1, 2, 2, 1], [-1, -1, 0, 0,0], [11, 10, 10, 12, 9]])
y = np.array([1, 0, 11])

# 调用 ReliefF 函数计算特征权重
k = 2  # 设置 k 值
weights = reliefF(X, y, k)

print("特征权重:", weights)