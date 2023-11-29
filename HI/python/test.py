# # # w = [0.3, 0.4, 0.5, 0.7, 0.01]
# # #
# # # # 找到w中的最小值和最大值
# # # min_value = min(w)
# # # max_value = max(w)
# # #
# # # # 如果最小值低于最大值的0.1倍，将最小值设置为0
# # # if min_value < 0.1 * max_value:
# # #     min_index = w.index(min_value)
# # #     w[min_index] = 0
# # #
# # # print(w)  # 打印更新后的w
# #
# # # import numpy as np
# # # from sklearn.ensemble import RandomForestClassifier
# # # from sklearn.feature_selection import RFE
# # #
# # # # 假设X是特征矩阵，y是标签向量
# # # X = np.random.rand(58, 5)  # 用随机数据代替
# # # y = np.random.randint(0, 2, 58)  # 用随机数据代替
# # #
# # # # 创建随机森林模型
# # # model = RandomForestClassifier()
# # #
# # # # 创建特征递归消除对象
# # # rfe = RFE(model, n_features_to_select=4)  # 选择3个最重要的特征
# # #
# # # # 使用特征递归消除选择特征
# # # rfe.fit(X, y)
# # #
# # # # 获取所选择的特征列索引值
# # # selected_feature_indices = np.where(rfe.ranking_ == 1)[0]
# # #
# # # print("选择的特征列索引值:", selected_feature_indices)
# #
# # # import numpy as np
# # #
# # # # 假设 x 是一个形状为 (58, 5) 的数组
# # # x = np.random.rand(58, 5)  # 示例数据，你可以使用你的实际数据
# # #
# # # # 计算两两列的对应元素相乘结果
# # # n_features = x.shape[1]
# # # combinations = [(i, j) for i in range(n_features) for j in range(i, n_features)]
# # # interaction_cols = [x[:, i] * x[:, j] for i, j in combinations]
# # #
# # # # 将原始列和相互作用列合并成新数组 X_
# # # X_ = np.column_stack([x] + interaction_cols)
# # #
# # # # X_ 包含原始列和两两列的对应元素相乘结果
# # # print(X_)
# #
# #
# # import numpy as np
# # import pandas as pd
# # from sklearn.ensemble import RandomForestRegressor
# # def rif_feature_importance(data_train,y_train, data_test, mtry):
# #     def random_forest_model(data_train, mtry):
# #         rf = RandomForestRegressor(n_estimators=100, max_features=mtry, random_state=1)
# #         rf.fit(data_train, y_train)
# #         return rf
# #
# #     rf_model = random_forest_model(data_train, mtry)
# #
# #     feature_importance = rf_model.feature_importances_
# #     feature_names = data_train.columns[:-1]  # Exclude the "class" column
# #
# #     feature_importance_dict = dict(zip(feature_names, feature_importance))
# #     sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
# #
# #     return sorted_feature_importance
# #
# #
# # # 示例特征和标签数据
# # features = np.random.rand(58, 5)  # 替换为您的特征数据
# # labels = np.random.randint(2, size=(58, 1))  # 替换为您的标签数据
# #
# # # 将特征和标签数据合并成一个DataFrame
# # data = pd.DataFrame(np.hstack((features, labels)), columns=[f'feature_{i}' for i in range(5)] + ['class'])
# #
# # # 划分数据集为训练集和测试集
# # np.random.seed(1)
# # train = np.random.choice(data.index, int(len(data) * 0.8), replace=False)
# # data_train = data.loc[train]
# # data_test = data.drop(train)
# #
# # mtry=4
# #
# # # 使用 rif_feature_importance 函数获取特征与标签之间的相关性大小
# # feature_importance = rif_feature_importance(data_train, data_test, mtry)
# # print("Feature Importance:")
# # for feature, importance in feature_importance:
# #     print(f"{feature}: {importance}")
# #
# # # # 运行随机森林模型并进行预测
# # # accuracy = rif_pred(data_train, data_test, mtry)
# # # print(f"Random Interaction Forest Accuracy: {accuracy}")
# #
# # import pywt
# # import numpy as np
# # import matplotlib.pyplot as plt
# #
# # # 创建一个示例信号
# # signal = np.random.rand(58, 6)
# # # signal = np.array([2, 4, 6, 8, 10, 8, 6, 4, 2, 0, -2, -4, -6, -8, -10, -8])
# #
# # # 小波分解
# # wavelet = "haar"  # 选择小波基函数，这里使用Haar小波
# # level = 2  # 分解的层数
# # coeffs = pywt.wavedec(signal, wavelet, level=level)
# #
# # # 提取逼近系数和细节系数
# # cA2, cD2, cD1 = coeffs
# #
# # # 重构信号
# # reconstructed_signal = pywt.waverec(coeffs, wavelet)
# #
# # # 绘制原始信号和重构信号
# # plt.figure(figsize=(10, 6))
# # plt.subplot(2, 1, 1)
# # plt.plot(signal, label="Original Signal")
# # plt.legend()
# # plt.subplot(2, 1, 2)
# # plt.plot(reconstructed_signal, label="Reconstructed Signal")
# # plt.legend()
# # plt.show()
#
# from PyEMD import CEEMDAN,EMD,EEMD
#
# def ceemdan(X_data):
#     IImfs = []
#     data = X_data.ravel()
#     ceemdan= CEEMDAN()
#     ceemdan.trials = 1  # 迭代次数
#     ceemdan.max_siftings = 18  # SIFT 迭代次数
#     ceemdan.noise_std = 9  # 白噪声标准差
#     ceemdan.ensemble_size=100
#
#     ceemdan.ceemdan(data)
#     imfs, res = ceemdan.get_imfs_and_residue()
#     plt.figure(figsize=(12, 9))
#     plt.subplots_adjust(hspace=0.1)
#     plt.subplot(imfs.shape[0] + 3, 1, 1)
#     plt.plot(data, 'r')
#     for i in range(imfs.shape[0]):
#         plt.subplot(imfs.shape[0] + 3, 1, i + 2)
#         plt.plot(imfs[i], 'g')
#         plt.ylabel("IMF %i" % (i + 1))
#         plt.locator_params(axis='x', nbins=10)
#         # 在函数前必须设置一个全局变量 IImfs=[]
#         IImfs.append(imfs[i])
#     plt.subplot(imfs.shape[0] + 3, 1, imfs.shape[0] + 3)
#     plt.plot(res, 'g')
#
#     return np.transpose(IImfs)
# def calculate(X,Y):
#     r,p=stats.spearmanr(X,Y)
#     return r
#
# from sklearn.decomposition import FastICA
# import pandas as pd
# import statsmodels.api as sm
# import matplotlib.pyplot as plt
# import numpy as np
# import scipy.stats as stats
# import math
#
# # 创建示例时间序列数据
# data = pd.read_csv('rowdata.csv')
# feature2 = np.array(data['HI100_150'])[84:]
# label = np.array(data['Cycle_life'])[84:]
# labellog = []
# for i in label:
#     labellog.append(math.log(i, 10))
#
# labellog = np.array(labellog).reshape(-1, 1)
# # index = pd.date_range(start='2023-01-01', periods=len(feature2), freq='D')
# # ts = pd.Series(feature2, index=index)
# #
# # # 执行时间序列分解
# # result = sm.tsa.seasonal_decompose(ts, model='additive')
# #
# # x = np.column_stack((ts.values, result.trend.values, result.seasonal.values, result.resid.values))
# # x= np.nan_to_num(x, nan=0.0)
# #
# # celllist = []
# # for col in range(len(x[0])):
# #     column_data = x[:, col]
# #     celllist.append(calculate(column_data, label))
#
# # # 绘制分解结果
# # fig, axes = plt.subplots(4, 1, figsize=(10, 8))
# # ts.plot(ax=axes[0], title='Original Time Series')
# # result.trend.plot(ax=axes[1], title='Trend Component')
# # result.seasonal.plot(ax=axes[2], title='Seasonal Component')
# # result.resid.plot(ax=axes[3], title='Residual Component')
# # plt.tight_layout()
# # plt.show()
#
# import numpy as np
# import pywt
# import matplotlib.pyplot as plt
#
# # import numpy as np
# #
# # # 假设 x 是一个包含 5 个不同长度列表的列表
# # x = [[1, 2, 3], [4, 5, 6, 7, 8], [9, 10, 11, 12], [13, 14], [15, 16, 17, 18, 19, 20]]
# #
# # # 创建一个空的形状为(40, 5)的数组
# # result = np.zeros((40, 5))
# #
# # # 对每个列表进行线性插值，扩展为长度为40的数组
# # for i, lst in enumerate(x):
# #     # 确定插值的位置
# #     xp = np.arange(len(lst))
# #     fp = np.array(lst)
# #     indices = np.linspace(0, len(lst) - 1, 40)
# #     result[:, i] = np.interp(indices, xp, fp)
# #
# # # result 现在包含了插值后的数组，每一列对应于一个插值后的数组
# # print(result)
#
# import math
#
# # 生成信号变量
# # t = np.linspace(0, 1, num=40)
# # signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 20 * t) + np.sin(3 * np.pi * 30 * t)
# # signal=feature2.ravel()
#
# # 添加随机噪声
# # noise = np.random.normal(0, 0.05, len(signal))
# # signal = signal + noise
#
# # 常见的几种小波基函数包括：
# #
# # # 1. Daubechies小波基（db）：Daubechies小波基是最常用的小波基函数之一。它具有紧凑支持和良好的频率局部化特性。常见的Daubechies小波基包括db2、db4、db6等。
# #
# # # 2. Symlets小波基（sym）：Symlets小波基是对称的Daubechies小波基。它们在频率局部化和相位对称性方面与Daubechies小波基类似。常见的Symlets小波基包括sym2、sym4、sym8等。
# #
# # # 3. Coiflets小波基（coif）：Coiflets小波基是具有紧凑支持和较好频率局部化特性的小波基。它们在一些应用中比Daubechies小波基具有更好的性能。常见的Coiflets小波基包括coif1、coif2、coif3等。
# #
# # # 4. Biorthogonal小波基（bior）：Biorthogonal小波基是一组成对的小波基函数。它们具有可变的支持长度和频率响应。常见的Biorthogonal小波基包括bior2.2、bior3.3、bior6.8等。
# #
# # # wavelet_name = 'db4'  # 定义小波基名称为'db4'
# # # wavelet_name = 'sym4'  # 定义小波基名称为'sym4'
# # wavelet_name = 'bior3.3'  # 定义小波基名称为'bior3.3'
# #
# # result = np.zeros((40, 5))
# # # 小波变换
# # coeffs = pywt.wavedec(signal, wavelet_name, level=4)  # 使用指定小波基进行4级小波分解
# # for i, lst in enumerate(coeffs):
# #     # 确定插值的位置
# #     xp = np.arange(len(lst))
# #     fp = np.array(lst)
# #     indices = np.linspace(0, len(lst) - 1, 40)
# #     result[:, i] = np.interp(indices, xp, fp)
# #
# #
# # # 绘制原始信号图像
# # plt.figure(figsize=(8, 6))
# # plt.subplot(5, 1, 1)
# # plt.plot(t, signal)
# # plt.title('Original Signal')
# # plt.xlabel('Time')
# # plt.ylabel('Amplitude')
# #
# #
# #
# # celllist = []
# # # 绘制小波分解信号图像
# # for i in range(len(result[0])):
# #     plt.subplot(5, 1, i + 1)
# #     plt.plot(t[:len(result[:,i])], result[:,i])
# #     print('gjbhnk')
# #     celllist.append(calculate(result[:,i], labellog))
# #     plt.title(f'Wavelet Coefficients - Level {i}')
# #     plt.xlabel('Time')
# #     plt.ylabel('Amplitude')
# #
# #
# #
# #
# # plt.tight_layout()
# # plt.show()
#
# from sklearn.decomposition import KernelPCA
# # ica = FastICA(n_components=1, random_state=0)  # 设置要提取的独立成分数量
# # S_ = ica.fit_transform(feature2.reshape(-1,1))  # S_ 中包含了分离后的独立成分
# #
# # # 将分离出的独立成分添加为数组的列
# # result_array = np.hstack((feature2.reshape(-1,1), S_))
#
# # 创建Kernel PCA模型并拟合数据
# # kpca = KernelPCA(kernel='linear', n_components=1)  # 使用径向基函数（RBF）核来处理非线性关系
# # X_kpca = kpca.fit_transform(feature2.reshape(-1,1))
# # result_array = np.hstack((feature2.reshape(-1,1), X_kpca))
#
#
# from pyswarm import pso
#
#
# # 定义CEEMDAN目标函数，根据CEEMDAN分解结果来评估效果
# from pyswarm import pso
#
#
# # 定义目标函数
# # def ceemdan_objective(parameters):
# #     trials, max_siftings, noise_std = parameters
# #     ceemdan_obj = CEEMDAN()
# #     ceemdan_obj.trials = trials
# #     ceemdan_obj.max_siftings = max_siftings
# #     ceemdan_obj.noise_std = noise_std
# #
# #     ceemdan_obj.ceemdan(data)
# #     imfs, _ = ceemdan_obj.get_imfs_and_residue()
# #
# #     # 这里可以根据需要定义目标函数，例如：最大化与标签的相关性
# #     # 请根据你的实际需求定义目标函数
# #
# #     # 这里假设目标函数为最大化 IMFs 的总能量
# #     total_energy = np.sum(np.square(imfs))
# #     return -total_energy  # PSO算法最小化目标函数，所以加负号来最大化能量
# #
# #
# # # 假设 data 是原始信号
# # data = np.random.rand(124)
# #
# # # 定义参数范围
# # parameter_ranges = [(50, 200), (10, 30), (5, 15)]  # trials, max_siftings, noise_std
# #
# # # 使用PSO算法搜索最佳参数
# # lb = [param[0] for param in parameter_ranges]  # 参数的下界
# # ub = [param[1] for param in parameter_ranges]  # 参数的上界
# #
# # best_params, _ = pso(ceemdan_objective, lb, ub, swarmsize=10, maxiter=100)
# #
# # # 获取最佳参数
# # trials, max_siftings, noise_std = best_params
# #
# # # 使用最佳参数运行CEEMDAN
# # ceemdan_obj = ceemdan.CEEMDAN()
# # ceemdan_obj.trials = trials
# # ceemdan_obj.max_siftings = max_siftings
# # ceemdan_obj.noise_std = noise_std
# # ceemdan_obj.ceemdan(data)
# # imfs, _ = ceemdan_obj.get_imfs_and_residue()
#
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import scipy.stats as stats
# from sklearn import svm
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# import xgboost
# import random
# from sklearn.decomposition import FastICA
# from sklearn.neural_network import MLPRegressor
# from math import sqrt
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_squared_error
# from xgboost import XGBRegressor
# from sklearn.decomposition import PCA
# from PyEMD import CEEMDAN,EMD,EEMD
# import statistics
# from sklearn.feature_selection import SelectFromModel
# from sklearn.ensemble import RandomForestRegressor
# from saved_xgb_regression_model import OptimizedXGBRegressor
# from sklearn.neighbors import NearestNeighbors
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import RFE
# import pywt
# import tensorflow as tf
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from tensorflow.python import keras
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam
# import statsmodels.api as sm
# def add_noise(arr, std_dev):
#     """
#     给定一个数组和标准差，返回添加了零均值标准差的噪声的新数组。
#
#     参数：
#     arr (numpy.ndarray)：输入数组
#     std_dev (float)：标准差（噪声的幅度）
#
#     返回：
#     numpy.ndarray：带有噪声的新数组
#     """
#     # 生成与输入数组相同形状的随机噪声
#     noise = np.random.normal(0, std_dev, arr.shape)
#
#     # 将噪声添加到输入数组中
#     noisy_arr = arr + noise
#
#     return noisy_arr
#
# def evaluation(y_test, y_predict):
#     mae = mean_absolute_error(y_test, y_predict)
#     mse = mean_squared_error(y_test, y_predict)
#     rmse = sqrt(mean_squared_error(y_test, y_predict))
#     return mae, rmse
#
#
# print(calculate(feature2, labellog))
#
# std = 0.001
# # feature2 = add_noise(feature2, std)#加噪声
# print(calculate(feature2, labellog))
# s_=ceemdan(feature2.reshape(-1,1))
# # result_array=feature2
# result_array = np.hstack((feature2.reshape(-1,1), s_))
#
# celllist = []
# for col in range(len(result_array[0])):
#     column_data = result_array[:, col]
#     celllist.append(calculate(column_data, labellog))
#
#
#
# print(celllist)
# X_train, X_test = result_array[:32,:], result_array[32:,:]
# y_train, y_test = labellog[:32], labellog[32:]
#
# xgb_model = XGBRegressor()
#
# # 拟合数据
# xgb_model.fit(X_train, y_train)
#
# # 预测
# y_pred = xgb_model.predict(X_test)
#
# mae, rmse = evaluation(y_test, y_pred)
# print('mae——{},rmse——{}'.format(mae, rmse))
#
#
#
#
# def rif_feature_importance(data_train, y_train, data_test, mtry):
#     def random_forest_model(data_train, y_train, mtry):
#         rf = RandomForestRegressor(n_estimators=100, max_features=mtry, random_state=1)
#         rf.fit(data_train, y_train)
#         return rf
#
#     rf_model = random_forest_model(data_train, y_train, mtry)
#
#     feature_importance = rf_model.feature_importances_
#
#     feature_importance_dict = {f'feature_{i}': importance for i, importance in enumerate(feature_importance)}
#     sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
#
#     return sorted_feature_importance
#
# def getRFE_RFfeatures(X,Y,X_test):
#     model = RandomForestRegressor()
#
#     # 创建特征递归消除对象
#     rfe = RFE(model, n_features_to_select=4)  # 选择3个最重要的特征
#
#     # 使用特征递归消除选择特征
#     rfe.fit(X, Y.ravel())
#
#     # 获取所选择的特征列索引值
#     selected_feature_indices = np.where(rfe.ranking_ == 1)[0]
#
#     print("选择的特征列索引值:", selected_feature_indices)
#
#     X_train=X[:,selected_feature_indices]
#     X__test=X_test[:,selected_feature_indices]
#
#     return X_train,X__test


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import SDA
# 定义配置
# class Config:
#     def __init__(self):
#         self.input_features = 5
#         self.hidden_features = [16, 8]  # 例如，有两个隐藏层，分别为 16 和 8 个特征
#         self.classes = 1
#         self.is_train = True
#         self.lr = 0.01
#         self.momentum = 0.9
#         self.weight_decay = 1e-5
#
# # 生成一些示例数据
# X_train = np.random.rand(32, 5)
# y_train = np.random.rand(32, 1)
#
# # 数据处理
# X_train = torch.tensor(X_train, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.float32)
#
# # 初始化模型
# config = Config()
# sda_model = SDA.SdA(config)
#
# # 定义损失函数和优化器
# criterion = nn.MSELoss()
# optimizer = optim.SGD(sda_model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
#
# # 训练模型
# num_epochs = 100
# for epoch in range(num_epochs):
#     # 前向传播
#     outputs = sda_model(X_train)
#
#     # 计算损失
#     loss = criterion(outputs, X_train)
#
#     # 反向传播和优化
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     # 打印训练信息
#     if (epoch + 1) % 10 == 0:
#         print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, loss.item()))
#
# # 在训练集上进行去噪
# denoised_data = X_train.detach().numpy()
# for layer in sda_model.layers[:-1]:
#     denoised_data = layer.encoder(torch.tensor(denoised_data, dtype=torch.float32)).detach().numpy()
#
# # 打印去噪后的数据
# print("Denoised Data:", denoised_data)
#
# arr=[0.2,-0.4,-0.8,0.8,0.9,0.4]
# selected_feature_indices = np.array(np.where(np.abs(arr)>0.5)).ravel()
# print(selected_feature_indices)
import pandas as pd

# 假设 df 是你的 DataFrame
# 用你的实际 DataFrame 替换这部分

# 示例 DataFrame
data = {
    'id_cycle': [1, 1, 2, 2, 3, 3],
    'value1': [10, 20, 30, 40, 50, 60],
    'value2': [15, 25, 35, 45, 55, 65]
}

df = pd.DataFrame(data)

# 计算每个 id_cycle 的均值
df_mean = df.groupby('id_cycle').mean().reset_index()

# 显示结果
print(df_mean)
