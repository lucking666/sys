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
from PyEMD import CEEMDAN
import statistics
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor


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


X_train, X_test = feature2_scale[:58, :], feature2_scale[58:84, :]
y_train, y_test = labellog[:58], labellog[58:84]






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
    ceemdan = CEEMDAN()
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
    X_train, X_test = np.array(np.transpose(IImfs))[:58, :], np.array(np.transpose(IImfs))[58:84, :]
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

    xgb = xgboost.XGBRegressor()
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)

    mae, rmse = evaluation(y_test, y_pred)
    # print('mae——{},rmse——{}'.format(mae, rmse))
    return mae, rmse


def getRFfeatures(X,Y,Xtest):
    clf = RandomForestRegressor(n_estimators=100)

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
    print("被选择的特征：", selected_features)
    X=selected_features
    selected_columns = Xtest[:, selected_feature_indices]


    return X,selected_columns



maelist = []
rmselist = []
maelistnoise = []
rmselistnoise = []
maelistnoiseemd = []
rmselistnoiseemd = []
maelistnoiseemd1 = []
rmselistnoiseemd1 = []
for i in range(1):
    random.seed(i)
    Xtrain = X_train
    Xtest = X_test
    mae, rmse = get_result(Xtrain, y_train, Xtest, y_test)
    std = random.uniform(0.01, 1)
    # std = 0.05
    Xtrain = add_noise(Xtrain, std)#加噪声
    Xtrain, Xtest = ceemdan(Xtrain, Xtest)
    Xtrain,Xtest=getRFfeatures(Xtrain,y_train,Xtest)
    print("到这里是模态分解完毕,使用随机森林进行特征选择,得到的结果作为最终结果")
    maenoiseemd, rmsenoiseemd = get_result(Xtrain, y_train, Xtest, y_test)
    maelistnoiseemd.append(maenoiseemd)
    rmselistnoiseemd.append(rmsenoiseemd)

print('加噪声emmae——{},rmse——{}'.format(np.median(maelistnoiseemd), np.median(rmselistnoiseemd)))

# 未打乱噪声随机（0.1-1）100
# 原始mae——0.06506055792111122,rmse——0.08479516256745924
# 加噪声mae——0.18514188773617876,rmse——0.22399060460510695
# 加噪声emmae——0.06313057499555567,rmse——0.08271343798051889
# 加噪声且变化emmae——0.05889135176690331,rmse——0.07248371396730052
# 加噪声emmae且加上——0.027635994847525804,rmse——0.04111405069986597


