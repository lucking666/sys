import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import svm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import xgboost
import random
from sklearn.decomposition import FastICA
from sklearn.neural_network import MLPRegressor
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from PyEMD import CEEMDAN,EMD,EEMD
import statistics
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from saved_xgb_regression_model import OptimizedXGBRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV,RFE
import pywt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.python import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import statsmodels.api as sm
from sklearn.decomposition import FastICA
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import sigmoid_kernel
from scipy.signal import find_peaks
from scipy.signal import find_peaks
from scipy.fftpack import fft, ifft
from numpy.fft import fftshift
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import copy
import usesdafordatatest2














rowdata = pd.read_csv('rowdata.csv')
random_data = rowdata
# random_data=rowdata.sample(frac=1).reset_index(drop=True).iloc[:,1:]

feature1 = np.array(random_data['HI10_100'])
feature2 = np.array(random_data['HI100_150'])
label = np.array(random_data['Cycle_life'])

# 归一化
from sklearn.preprocessing import MinMaxScaler


def normalization(data):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(np.array(data).reshape(-1, 1))  #
    return data


feature1_scale = normalization(feature1)
feature2_scale = normalization(feature2)

# 对特征和数据进行取对数
import math

labellog = []
for i in label:
    labellog.append(math.log(i, 10))

labellog = np.array(labellog).reshape(-1, 1)


# labellog=np.array(label).reshape(-1, 1)


feature1_scale = feature1_scale[:]
feature2_scale = feature2_scale[:]
feature_scale = np.concatenate([feature1_scale, feature2_scale], axis=1)

# Spearman系数


# def calculate_spearman_correlation(X, Y):
#     r,p=stats.spearmanr(X, Y)
#     return r,p
#
# r,p=calculate_spearman_correlation(feature1,label)
# print("相关系数：",r)

# feature1 feature2相关系数
# -0.7842766999664926,-0.899444523235248
# feature1_scale feature2_scale相关系数
# -0.7842766999664926,-0.899444523235248
# feature1log feature2log相关系数
# -0.7842766999664926,-0.899444523235248


X_train, X_test = feature2_scale[84:116,:], feature2_scale[116:,:]
y_train, y_test = labellog[84:116], labellog[116:]



# def interaction_subtract(x):
#     n_features = x.shape[1]
#
#     # 初始化一个空的列表来存储相减列
#     subtracted_cols = []
#
#     # 使用两个嵌套循环来计算每一对特征之间的差值
#     for i in range(n_features):
#         for j in range(i, n_features):
#             subtracted_col = x[:, i] - x[:, j]
#             subtracted_cols.append(subtracted_col)
#
#     # 将原始列和相减列合并成新数组 X_
#     X_ = np.column_stack([x] + subtracted_cols)
#
#     return X_
def interaction_subtract(x):
    n_features = x.shape[1]

    # 初始化一个空的列表来存储相减列
    subtracted_cols = []

    # 使用两个嵌套循环来计算每一对特征之间的差值
    for i in range(1,n_features):
        subtracted_col = x[:, 0] - x[:, i]
        subtracted_cols.append(subtracted_col)

    # 将原始列和相减列合并成新数组 X_
    X_ = np.column_stack([x] + subtracted_cols)

    return X_

def add_noise(arr, std_dev,index):
    """
    给定一个数组和标准差，返回添加了零均值标准差的噪声的新数组。

    参数：
    arr (numpy.ndarray)：输入数组
    std_dev (float)：标准差（噪声的幅度）

    返回：
    numpy.ndarray：带有噪声的新数组
    """
    # 生成与输入数组相同形状的随机噪声
    np.random.seed(0)
    noise = np.random.normal(0, std_dev, arr.shape)
    # print(noise)

    # 将噪声添加到输入数组中
    noisy_arr = arr + noise

    return noisy_arr


# 加噪声

def _SDA(X_train,y_train,X_test):
    # 示例用法
    config = usesdafordatatest2.Config()

    denoised_X_train, denoised_X_test = usesdafordatatest2.train_and_denoise(X_train, y_train, X_test, config)

    return denoised_X_train,denoised_X_test


# def ceemdan(X_train, X_test,t):
#     X_data = np.vstack((X_train, X_test))
#     IImfs = []
#     data = copy.deepcopy(X_data).ravel()
#     ceemdan= CEEMDAN()
#     ceemdan.trials = 100  # 迭代次数
#     ceemdan.max_siftings = 50  # SIFT 迭代次数
#     ceemdan.noise_std = 0.01  # 白噪声标准差
#     ceemdan.ensemble_size=5
#
#     ceemdan.ceemdan(data)
#     imfs, res = ceemdan.get_imfs_and_residue()
#     # plt.figure(figsize=(12, 9))
#     # plt.subplots_adjust(hspace=0.1)
#     # plt.subplot(imfs.shape[0] + 3, 1, 1)
#     # plt.plot(data, 'r')
#     for i in range(imfs.shape[0]):
#         # plt.subplot(imfs.shape[0] + 3, 1, i + 2)
#         # plt.plot(imfs[i], 'g')
#         # plt.ylabel("IMF %i" % (i + 1))
#         # plt.locator_params(axis='x', nbins=10)
#         # 在函数前必须设置一个全局变量 IImfs=[]
#         IImfs.append(imfs[i])
#     # plt.subplot(imfs.shape[0] + 3, 1, imfs.shape[0] + 3)
#     # plt.plot(res, 'g')
#     new_data=result_array=np.hstack((t,X_data, np.transpose(IImfs)))
#     X_train, X_test = new_data[:len(X_train), :], new_data[len(X_train):, :]
#     X_train = interaction_subtract(X_train)
#     X_test = interaction_subtract(X_test)
#     return X_train, X_test

def ceemdan(X_train, X_test):
    X_data = np.vstack((X_train, X_test))
    IImfs = []
    data = copy.deepcopy(X_data).ravel()
    ceemdan= CEEMDAN()
    ceemdan.trials = 1  # 迭代次数
    ceemdan.max_siftings = 1  # SIFT 迭代次数
    ceemdan.noise_std = 0.605 # 白噪声标准差
    ceemdan.ensemble_size=1

    ceemdan.ceemdan(data)
    imfs, res = ceemdan.get_imfs_and_residue()
    # plt.figure(figsize=(12, 9))
    # plt.subplots_adjust(hspace=0.1)
    # plt.subplot(imfs.shape[0] + 3, 1, 1)
    # plt.plot(data, 'r')
    for i in range(imfs.shape[0]):
        # plt.subplot(imfs.shape[0] + 3, 1, i + 2)
        # plt.plot(imfs[i], 'g')
        # plt.ylabel("IMF %i" % (i + 1))
        # plt.locator_params(axis='x', nbins=10)
        # 在函数前必须设置一个全局变量 IImfs=[]
        IImfs.append(imfs[i])
    # plt.subplot(imfs.shape[0] + 3, 1, imfs.shape[0] + 3)
    # plt.plot(res, 'g')
    result_array=np.transpose(IImfs)
    new_data=np.hstack((X_data, np.transpose(IImfs)))
    X_train, X_test = new_data[:len(X_train), :], new_data[len(X_train):, :]
    X_train = interaction_subtract(X_train)
    X_test = interaction_subtract(X_test)
    # X_train = np.delete(X_train, np.where(~X_train.any(axis=0))[0], axis=1)
    # X_test = np.delete(X_test, np.where(~X_test.any(axis=0))[0], axis=1)
    return X_train, X_test

def Pywt(X_train, X_test):
    X_data = np.vstack((X_train, X_test))

    wavelet =  "db8"
    level = 6
    # 进行小波分解
    coeffs = pywt.wavedec(X_data, wavelet, level=level)



    # 重构数据
    X_data_reconstructed = pywt.waverec(coeffs, wavelet)

    # 拆分重构后的数据回到训练集和测试集
    X_train_reconstructed = X_data_reconstructed[:len(X_train), :]
    X_test_reconstructed = X_data_reconstructed[len(X_train):, :]

    return X_train_reconstructed, X_test_reconstructed




def seasonal(X_train, X_test):
    # 将 X_train 和 X_test 合并成一个大的数据集 X_data
    X_data = np.vstack((X_train, X_test))

    # 初始化一个列表来保存分解后的成分
    seasonal_components = []
    index = pd.date_range(start='2023-01-01', periods=len(X_data), freq='D')
    # 创建一个 Pandas Series 以便进行季节性分解
    ts = pd.Series(X_data.ravel(), index=index)  # 选择 X_data 的第一列进行分解

    # 执行时间序列分解
    result = sm.tsa.seasonal_decompose(ts, model='additive')

    # 提取分解结果中的各个成分，并将它们添加到 seasonal_components 列表中
    seasonal_components = np.column_stack((ts.values, result.trend.values, result.seasonal.values, result.resid.values))
    seasonal_components= np.nan_to_num(seasonal_components, nan=0.0)


    # 分割分解后的数据，以便返回 X_train 和 X_test
    X_train, X_test = seasonal_components[:len(X_train), :], seasonal_components[len(X_train):, :]

    return X_train, X_test


def pca(X_train, X_test):
    # 将 X_train 和 X_test 合并成一个大的数据集 X_data
    X_data = np.vstack((X_train, X_test))

    # ica = FastICA(n_components=1, random_state=0)  # 设置要提取的独立成分数量
    # S_ = ica.fit_transform(X_data)  # S_ 中包含了分离后的独立成分
    #
    # # 将分离出的独立成分添加为数组的列
    # result_array = np.hstack((X_data, S_))

    n_components = 2  # 指定要提取的独立成分数量
    ica = FastICA(n_components=n_components, max_iter=2000, random_state=0)

    # # 创建线性核
    # X_linear_kernel = linear_kernel(X_data)

    # 创建Sigmoid核
    gamma = 0.1  # 核函数的参数
    coef0 = 1  # 偏置项的系数
    X_sigmoid_kernel = sigmoid_kernel(X_data, gamma=gamma, coef0=coef0)

    # # 创建多项式核
    # degree = 3  # 多项式的次数
    # coef0 = 1  # 线性项的系数
    # X_poly_kernel = polynomial_kernel(X_data, degree=degree, coef0=coef0)
    # 使用核函数（如径向基函数 RBF 核）将数据映射到高维空间
    # gamma = 1.0  # 核函数参数
    # X_kernel = rbf_kernel(X_data, gamma=gamma)

    # 执行非线性ICA分解
    ica_components = ica.fit_transform(X_sigmoid_kernel)

    # ica_components 包含了提取的非线性独立成分

    # 还可以获取混合矩阵，以便对新数据进行转换
    mixing_matrix = ica.mixing_

    result_array=np.hstack((X_data, ica_components))

    # # 创建PCA模型并拟合数据
    # pca = PCA(n_components=1)  # 设置要提取的主成分数量
    # X_pca = pca.fit_transform(X_data)  # X_pca 中包含了主成分
    #
    # # 将分离出的主成分和原始特征添加为数组的两列
    # result_array = np.hstack((X_data, X_pca))

    # 分割分解后的数据，以便返回 X_train 和 X_test
    X_train, X_test = result_array[:len(X_train), :], result_array[len(X_train):, :]

    return X_train, X_test



def lmd(X_train, X_test):
    # 将 X_train 和 X_test 合并成一个大的数据集 X_data
    X_data = np.vstack((X_train, X_test))

    def local_mean_decomposition_array(input_array):
        # 将输入数组转换为一维
        signal = input_array.ravel()
        # 找到信号的局部极值点
        peaks, _ = find_peaks(signal)
        # 计算局部均值
        local_means = []
        for i in range(len(peaks) - 1):
            local_mean = np.mean(signal[peaks[i]:peaks[i + 1]])
            local_means.append(local_mean)
        # 计算局部成分
        local_components = signal - np.interp(np.arange(len(signal)), peaks[:-1], local_means)
        # 将局部成分重新整理为形状 (40, n)
        n = len(local_components)
        local_components = local_components.reshape(-1, 1)
        return local_components



    components=local_mean_decomposition_array(X_data)

    result_array=np.hstack((X_data, components))

    X_train, X_test = result_array[:len(X_train), :], result_array[len(X_train):, :]

    return X_train, X_test

def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mean_squared_error(y_test, y_predict))
    return mae, rmse


def get_result(X_train, y_train, X_test, y_test):
    # model1=svm.SVR(probability = True,kernel = 'rbf',c=0.1,max_iter=10)
    # model1.fit(X_train,y_train)
    # y_pred=model1.predict(X_test)

    # lr=LinearRegression().fit(X_train,y_train)
    # y_pred=lr.predict(X_test)

    # 创建OptimizedXGBRegressor对象
    xgb_model = XGBRegressor()

    # 拟合数据
    xgb_model.fit(X_train, y_train)

    # 预测
    y_pred = xgb_model.predict(X_test)

    mae, rmse = evaluation(y_test, y_pred)
    # print('mae——{},rmse——{}'.format(mae, rmse))
    return mae, rmse


def get_result0(X, y, X_test, y_test):
    # model1=svm.SVR(probability = True,kernel = 'rbf',c=0.1,max_iter=10)
    # model1.fit(X_train,y_train)
    # y_pred=model1.predict(X_test)

    # lr=LinearRegression().fit(X_train,y_train)
    # y_pred=lr.predict(X_test)

    # 创建OptimizedXGBRegressor对象
    # # 使用SVR进行预测
    # svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    # svr_lin = SVR(kernel='linear', C=100, epsilon=0.1)
    # svr_poly = SVR(kernel='poly', C=100, degree=3, epsilon=0.1)
    #
    # # 拟合模型
    # y_rbf = svr_rbf.fit(X, y).predict(X_test)
    # y_lin = svr_lin.fit(X, y).predict(X_test)
    # y_poly = svr_poly.fit(X, y).predict(X_test)
    #
    #
    # mae, rmse = evaluation(y_test, y_rbf)
    # mae1, rmse1 = evaluation(y_test, y_rbf)
    # mae2, rmse2 = evaluation(y_test, y_rbf)

    regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=0)

    # 拟合模型
    regr.fit(X, y)

    # 预测
    y_pred = regr.predict(X_test)
    mae, rmse = evaluation(y_test, y_pred)
    # print('mae——{},rmse——{}'.format(mae, rmse))
    return mae, rmse


def train_and_evaluate_nn(Xtrain, y_train, Xtest, y_test):
    # 创建一个简单的前馈神经网络模型
    model = Sequential()
    model.add(Dense(64, input_dim=Xtrain.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # 输出层，通常用于回归任务

    # 编译模型
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))  # 使用均方误差作为损失函数

    # 训练模型
    model.fit(Xtrain, y_train, epochs=100, batch_size=32, verbose=0)  # 你可以调整训练的参数

    # 在测试数据上进行预测
    y_pred = model.predict(Xtest)

    # 计算 MAE 和 RMSE
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return mae, rmse








def showFeatures(data):
    x = range(len(data))

    # 创建纵轴数据，使用data的5个列
    y = data

    # 创建图形
    plt.plot(x, y)

    # 添加标题和标签
    plt.title('Data Plot')
    plt.xlabel('横轴')
    plt.ylabel('纵轴')

    # 显示图形
    plt.show()
def getRFfeatures(X,Y,Xtest):
    clf = RandomForestRegressor(n_estimators=100)

    # for index in range(len(X[0])):
    #     data=X[:,index]
    #     showFeatures(data)

    # 使用训练集X和Y进行拟合
    clf.fit(X, Y)

    # 使用SelectFromModel进行特征选择
    sfm = SelectFromModel(clf, threshold='mean')
    sfm.fit(X, Y)

    # 获取特征的重要性分数
    feature_importances = clf.feature_importances_
    # 选择前n个最重要的特征
    n = 2  # 选择前3个特征，你可以根据需要调整n的值
    temp=np.argsort(feature_importances)[::-1]
    selected_feature_indices = temp[:n]
    # selected_feature_indices = np.array(np.where(np.abs(feature_importances)>0.5)).ravel()
    # 打印被选择的特征的索引和特征值
    print("被选择的特征的索引：", selected_feature_indices)
    selected_features = X[:, selected_feature_indices]
    # print("被选择的特征：", selected_features)
    X=selected_features
    selected_columns = Xtest[:, selected_feature_indices]


    return X,selected_columns,selected_feature_indices

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


def getReliefFfeatures(X,Y,X_test):
    k = 2  # 设置 k 值
    weights = reliefF(X, Y, k)


    w=weights
    # w=np.abs(weights)
    for i in range(len(weights)):
        if weights[i]<0.1:
            weights[i]=0

    print("特征权重:", weights)

    X_train=X*w
    X__test=X_test*w


    return X_train,X__test


def getRFE_RFfeatures(X,Y,X_test):
    model = RandomForestRegressor()

    # 创建特征递归消除对象
    # rfe = RFECV(model,min_features_to_select=1,)
    rfe = RFE(model, n_features_to_select=2 )


    # 使用特征递归消除选择特征
    rfe.fit(X, Y.ravel())

    # 获取所选择的特征列索引值
    selected_feature_indices = np.where(rfe.ranking_ == 1)[0]


    print("选择的特征列索引值:", selected_feature_indices)

    X_train=X[:,selected_feature_indices]
    X__test=X_test[:,selected_feature_indices]


    return X_train,X__test

import scipy.stats as stats
def calculate(X,Y):
    r,p=stats.spearmanr(X,Y)
    return r


def rif_feature_importance(data_train, y_train, data_test, mtry):
    def random_forest_model(data_train, y_train, mtry):
        rf = RandomForestRegressor(n_estimators=100, max_features=mtry, random_state=1)
        rf.fit(data_train, y_train)
        return rf

    rf_model = random_forest_model(data_train, y_train, mtry)

    feature_importance = rf_model.feature_importances_

    feature_importance_dict = {f'feature_{i}': importance for i, importance in enumerate(feature_importance)}
    sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    return sorted_feature_importance



maelist = []
rmselist = []
maelistnoise = []
rmselistnoise = []
maelistnoiseemd = []
rmselistnoiseemd = []
maelistnoiseemd1 = []
rmselistnoiseemd1 = []
for i in range(10):
    random.seed(i)
    t=np.vstack((X_train, X_test))
    Xtrain = X_train
    Xtest = X_test
    mae, rmse = get_result(Xtrain, y_train, Xtest, y_test)

    spman = calculate(Xtrain, y_train)#未加噪声
    std = random.uniform(0.001, 0.1)


    std =0.005
    # print(std)
    # Xtrain = add_noise(Xtrain, std,i)#加噪声
    maenoise, rmsenoise = get_result(Xtrain, y_train, Xtest, y_test)
    maelistnoise.append(maenoise)
    rmselistnoise.append(rmsenoise)
    spman=calculate(Xtrain,y_train)

    # Xtrain, Xtest = Pywt(Xtrain, Xtest)

    # Xtrain, Xtest = ceemdan(Xtrain, Xtest,t)
    # Xtrain, Xtest = _SDA(Xtrain, y_train, Xtest)

    Xtrain, Xtest = ceemdan(Xtrain, Xtest)

    # Xtrain, Xtest = seasonal(Xtrain, Xtest)
    # Xtrain, Xtest = pca(Xtrain, Xtest)
    # Xtrain, Xtest =  lmd(Xtrain, Xtest)

    celllist = []
    for col in range(len(Xtrain[0])):
        column_data = Xtrain[:, col]
        celllist.append(calculate(column_data, y_train))
    print(celllist)
    #使用随机交互森林对特征进行评估
    # feature_importance = rif_feature_importance(Xtrain,y_train, Xtest, 5)
    # print("Feature Importance:")
    # for feature, importance in feature_importance:
    #     print(f"{feature}: {importance}")

    Xtrain,Xtest,index=getRFfeatures(Xtrain,y_train,Xtest)#随机森林
    # Xtrain=np.delete(Xtrain, index, axis=1)
    # Xtest=np.delete(Xtest,index,axis=1)

    # Xtrain, Xtest = getReliefFfeatures(Xtrain, y_train, Xtest)#reliefF算法
    Xtrain, Xtest = getRFE_RFfeatures(Xtrain, y_train, Xtest)#特征递归消除和随机森林结合

    # Xtrain=np.hstack((Xtrain__,Xtrain))
    # Xtest=np.hstack((Xtest__,Xtest))

    # maenoiseemd, rmsenoiseemd = train_and_evaluate_nn(Xtrain, y_train, Xtest, y_test)
    maenoiseemd, rmsenoiseemd = get_result(Xtrain, y_train, Xtest, y_test)
    maelist.append(mae)
    rmselist.append(rmse)
    maelistnoiseemd.append(maenoiseemd)
    rmselistnoiseemd.append(rmsenoiseemd)
    print(std,maenoiseemd,rmsenoiseemd)

print(maelistnoiseemd,rmselistnoiseemd)
print('原始mae——{},rmse——{}'.format(np.median(maelist), np.median(rmselist)))
print('加噪声mae——{},rmse——{}'.format(np.median(maelistnoise), np.median(rmselistnoise)))
print('加噪声加特征选择算法mae——{},rmse——{}'.format(np.median(maelistnoiseemd), np.median(rmselistnoiseemd)))




# logging
# 原始mae——0.09068857431951866,rmse——0.12329421229404494
# 加噪声mae——0.11363843872574869,rmse——0.14422112406943496
# 加噪声em加特征选择算法mae——0.08149184805587845,rmse——0.11195285068127579

# 选择分量之间不作差
# 原始mae——0.09068857431951866,rmse——0.12329421229404494
# 加噪声mae——0.13573925716117935,rmse——0.1624299358420838
# 加噪声加特征选择算法mae——0.12013542846033021,rmse——0.1443557576036395