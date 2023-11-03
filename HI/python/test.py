# # w = [0.3, 0.4, 0.5, 0.7, 0.01]
# #
# # # 找到w中的最小值和最大值
# # min_value = min(w)
# # max_value = max(w)
# #
# # # 如果最小值低于最大值的0.1倍，将最小值设置为0
# # if min_value < 0.1 * max_value:
# #     min_index = w.index(min_value)
# #     w[min_index] = 0
# #
# # print(w)  # 打印更新后的w
#
# # import numpy as np
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.feature_selection import RFE
# #
# # # 假设X是特征矩阵，y是标签向量
# # X = np.random.rand(58, 5)  # 用随机数据代替
# # y = np.random.randint(0, 2, 58)  # 用随机数据代替
# #
# # # 创建随机森林模型
# # model = RandomForestClassifier()
# #
# # # 创建特征递归消除对象
# # rfe = RFE(model, n_features_to_select=4)  # 选择3个最重要的特征
# #
# # # 使用特征递归消除选择特征
# # rfe.fit(X, y)
# #
# # # 获取所选择的特征列索引值
# # selected_feature_indices = np.where(rfe.ranking_ == 1)[0]
# #
# # print("选择的特征列索引值:", selected_feature_indices)
#
# # import numpy as np
# #
# # # 假设 x 是一个形状为 (58, 5) 的数组
# # x = np.random.rand(58, 5)  # 示例数据，你可以使用你的实际数据
# #
# # # 计算两两列的对应元素相乘结果
# # n_features = x.shape[1]
# # combinations = [(i, j) for i in range(n_features) for j in range(i, n_features)]
# # interaction_cols = [x[:, i] * x[:, j] for i, j in combinations]
# #
# # # 将原始列和相互作用列合并成新数组 X_
# # X_ = np.column_stack([x] + interaction_cols)
# #
# # # X_ 包含原始列和两两列的对应元素相乘结果
# # print(X_)
#
#
# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# def rif_feature_importance(data_train,y_train, data_test, mtry):
#     def random_forest_model(data_train, mtry):
#         rf = RandomForestRegressor(n_estimators=100, max_features=mtry, random_state=1)
#         rf.fit(data_train, y_train)
#         return rf
#
#     rf_model = random_forest_model(data_train, mtry)
#
#     feature_importance = rf_model.feature_importances_
#     feature_names = data_train.columns[:-1]  # Exclude the "class" column
#
#     feature_importance_dict = dict(zip(feature_names, feature_importance))
#     sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
#
#     return sorted_feature_importance
#
#
# # 示例特征和标签数据
# features = np.random.rand(58, 5)  # 替换为您的特征数据
# labels = np.random.randint(2, size=(58, 1))  # 替换为您的标签数据
#
# # 将特征和标签数据合并成一个DataFrame
# data = pd.DataFrame(np.hstack((features, labels)), columns=[f'feature_{i}' for i in range(5)] + ['class'])
#
# # 划分数据集为训练集和测试集
# np.random.seed(1)
# train = np.random.choice(data.index, int(len(data) * 0.8), replace=False)
# data_train = data.loc[train]
# data_test = data.drop(train)
#
# mtry=4
#
# # 使用 rif_feature_importance 函数获取特征与标签之间的相关性大小
# feature_importance = rif_feature_importance(data_train, data_test, mtry)
# print("Feature Importance:")
# for feature, importance in feature_importance:
#     print(f"{feature}: {importance}")
#
# # # 运行随机森林模型并进行预测
# # accuracy = rif_pred(data_train, data_test, mtry)
# # print(f"Random Interaction Forest Accuracy: {accuracy}")

import pywt
import numpy as np
import matplotlib.pyplot as plt

# 创建一个示例信号
signal = np.random.rand(58, 6)
# signal = np.array([2, 4, 6, 8, 10, 8, 6, 4, 2, 0, -2, -4, -6, -8, -10, -8])

# 小波分解
wavelet = "haar"  # 选择小波基函数，这里使用Haar小波
level = 2  # 分解的层数
coeffs = pywt.wavedec(signal, wavelet, level=level)

# 提取逼近系数和细节系数
cA2, cD2, cD1 = coeffs

# 重构信号
reconstructed_signal = pywt.waverec(coeffs, wavelet)

# 绘制原始信号和重构信号
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(signal, label="Original Signal")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(reconstructed_signal, label="Reconstructed Signal")
plt.legend()
plt.show()
