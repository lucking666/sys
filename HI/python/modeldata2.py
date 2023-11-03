import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import svm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import xgboost
import random
from sklearn.neural_network import MLPRegressor
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from PyEMD import CEEMDAN,EMD,EEMD
import statistics
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from saved_xgb_regression_model import OptimizedXGBRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import pywt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.python import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


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






def add_noise(arr, std_dev):
    """
    给定一个数组和标准差，返回添加了零均值标准差的噪声的新数组。

    参数：
    arr (numpy.ndarray)：输入数组
    std_dev (float)：标准差（噪声的幅度）

    返回：
    numpy.ndarray：带有噪声的新数组
    """
    # 生成与输入数组相同形状的随机噪声
    noise = np.random.normal(0, std_dev, arr.shape)

    # 将噪声添加到输入数组中
    noisy_arr = arr + noise

    return noisy_arr


# 加噪声

def ceemdan(X_train, X_test):
    X_data = np.vstack((X_train, X_test))
    IImfs = []
    data = X_data.ravel()
    eemd= CEEMDAN()
    eemd.ensemble_size=5

    eemd.ceemdan(data)
    imfs, res = eemd.get_imfs_and_residue()
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
    X_train, X_test = np.transpose(IImfs)[:32, :], np.transpose(IImfs)[32:, :]
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
    X_train_reconstructed = X_data_reconstructed[:32, :]
    X_test_reconstructed = X_data_reconstructed[32:, :]

    return X_train_reconstructed, X_test_reconstructed






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
    xgb_model = OptimizedXGBRegressor()

    # 拟合数据
    xgb_model.fit(X_train, y_train)

    # 预测
    y_pred = xgb_model.predict(X_test)

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

def interaction(x):
    n_features = x.shape[1]
    combinations = [(i, j) for i in range(n_features) for j in range(i, n_features)]
    interaction_cols = [x[:, i] * x[:, j] for i, j in combinations]

    # 将原始列和相互作用列合并成新数组 X_
    X_ = np.column_stack([x] + interaction_cols)

    return X_



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
    # X =  interaction(X)
    # Xtest = interaction(Xtest)
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
    n = 3  # 选择前3个特征，你可以根据需要调整n的值
    selected_feature_indices = np.argsort(feature_importances)[::-1][:n]
    selected_features = X[:, selected_feature_indices]

    # 打印被选择的特征的索引和特征值
    print("被选择的特征的索引：", selected_feature_indices)
    # print("被选择的特征：", selected_features)
    X=selected_features
    selected_columns = Xtest[:, selected_feature_indices]


    return X,selected_columns

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
    rfe = RFE(model, n_features_to_select=4)  # 选择3个最重要的特征

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
for i in range(40):
    random.seed(i)
    Xtrain = X_train
    Xtest = X_test
    mae, rmse = get_result(Xtrain, y_train, Xtest, y_test)

    spman = calculate(Xtrain, y_train)#未加噪声0.74
    # std = random.uniform(0.01, 1)
    std = 0.8
    Xtrain = add_noise(Xtrain, std)#加噪声
    maenoise, rmsenoise = get_result(Xtrain, y_train, Xtest, y_test)
    maelistnoise.append(maenoise)
    rmselistnoise.append(rmsenoise)
    spman=calculate(Xtrain,y_train)#加噪声之后0.58

    # Xtrain, Xtest = Pywt(Xtrain, Xtest)

    Xtrain, Xtest = ceemdan(Xtrain, Xtest)
    celllist = []
    for col in range(len(Xtrain[0])):
        column_data = Xtrain[:, col]
        celllist.append(calculate(column_data, y_train))
    #使用随机交互森林对特征进行评估
    # feature_importance = rif_feature_importance(Xtrain,y_train, Xtest, 5)
    # print("Feature Importance:")
    # for feature, importance in feature_importance:
    #     print(f"{feature}: {importance}")

    # Xtrain,Xtest=getRFfeatures(Xtrain,y_train,Xtest)#随机森林
    # Xtrain, Xtest = getReliefFfeatures(Xtrain, y_train, Xtest)#reliefF算法
    # Xtrain, Xtest = getRFE_RFfeatures(Xtrain, y_train, Xtest)#特征递归消除和随机森林结合
    # print("到这里是模态分解完毕,使用随机森林进行特征选择,得到的结果作为最终结果")
    maenoiseemd, rmsenoiseemd = train_and_evaluate_nn(Xtrain, y_train, Xtest, y_test)
    # maenoiseemd, rmsenoiseemd = get_result(Xtrain, y_train, Xtest, y_test)
    maelist.append(mae)
    rmselist.append(rmse)
    maelistnoiseemd.append(maenoiseemd)
    rmselistnoiseemd.append(rmsenoiseemd)

print('原始mae——{},rmse——{}'.format(np.median(maelist), np.median(rmselist)))
print('加噪声mae——{},rmse——{}'.format(np.median(maelistnoise), np.median(rmselistnoise)))
print('加噪声em加特征选择算法mae——{},rmse——{}'.format(np.median(maelistnoiseemd), np.median(rmselistnoiseemd)))



# 二次测试数据集：（未log）
# 原始mae——151.8452377319336,rmse——179.58825131998603
# 加噪声mae——358.31847763061523,rmse——463.7400078847194
# 随机森林4个特征：加噪声em加特征选择算法mae——333.8544044494629,rmse——440.30145763007016
# 随机森林和递归消除四个：加噪声em加特征选择算法mae——329.8651924133301,rmse——450.208018318938
# reliefF算法：加噪声em加特征选择算法mae——342.79841232299805,rmse——458.81434883408366

# logging
# 原始mae——0.06398691535536155,rmse——0.07855878847070456
# 加噪声mae——0.1393503760683043,rmse——0.18856930717886522
# 加噪声em加特征选择算法mae——0.12864023210057018,rmse——0.15867174959175068reliefF
# 加噪声em加特征选择算法mae——0.1274263818656423,rmse——0.15203856024825693随机森林4特征
# 加噪声em加特征选择算法mae——0.12379297229481223,rmse——0.15590520682781867RFE_RF