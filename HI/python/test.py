# w = [0.3, 0.4, 0.5, 0.7, 0.01]
#
# # 找到w中的最小值和最大值
# min_value = min(w)
# max_value = max(w)
#
# # 如果最小值低于最大值的0.1倍，将最小值设置为0
# if min_value < 0.1 * max_value:
#     min_index = w.index(min_value)
#     w[min_index] = 0
#
# print(w)  # 打印更新后的w

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

# 假设X是特征矩阵，y是标签向量
X = np.random.rand(58, 5)  # 用随机数据代替
y = np.random.randint(0, 2, 58)  # 用随机数据代替

# 创建随机森林模型
model = RandomForestClassifier()

# 创建特征递归消除对象
rfe = RFE(model, n_features_to_select=4)  # 选择3个最重要的特征

# 使用特征递归消除选择特征
rfe.fit(X, y)

# 获取所选择的特征列索引值
selected_feature_indices = np.where(rfe.ranking_ == 1)[0]

print("选择的特征列索引值:", selected_feature_indices)



