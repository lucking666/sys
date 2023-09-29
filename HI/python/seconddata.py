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
    a=np.transpose(IImfs)
    X_train, X_test = np.transpose(IImfs)[:32, :], np.transpose(IImfs)[32:, :]
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

    # gpr = GaussianProcessRegressor()
    # gpr.fit(X_train,y_train)
    # y_pred=gpr.predict(X_test)

    # mlp=MLPRegressor(hidden_layer_sizes = (500,500,500))
    # mlp.fit(X_train,y_train)
    # y_pred=mlp.predict(X_test)

    mae, rmse = evaluation(y_test, y_pred)
    # print('mae——{},rmse——{}'.format(mae, rmse))
    return mae, rmse


# X_train=add_noise(X_train,0.05)
# X_train,X_test=ceemdan(X_train,X_test)

maelist = []
rmselist = []
maelistnoise = []
rmselistnoise = []
maelistnoiseemd = []
rmselistnoiseemd = []

for i in range(100):
    Xtrain=X_train
    Xtest=X_test
    mae,rmse=get_result(Xtrain, y_train, Xtest, y_test)
    maelist.append(mae)
    rmselist.append(rmse)
    # std=random.uniform(0.01, 0.2)
    std=0.05
    Xtrain = add_noise(Xtrain, std)
    maenoise, rmsenoise = get_result(Xtrain, y_train, Xtest, y_test)
    maelistnoise.append(maenoise)
    rmselistnoise.append(rmsenoise)
    Xtrain, Xtest = ceemdan(Xtrain, Xtest)
    maenoiseemd, rmsenoiseemd = get_result(Xtrain, y_train, Xtest, y_test)
    maelistnoiseemd.append(maenoiseemd)
    rmselistnoiseemd.append(rmsenoiseemd)

print('原始mae——{},rmse——{}'.format(np.mean(maelist), np.mean(rmselist)))
print('加噪声mae——{},rmse——{}'.format(np.mean(maelistnoise), np.mean(rmselistnoise)))
print('加噪声emmae——{},rmse——{}'.format(np.mean(maelistnoiseemd), np.mean(rmselistnoiseemd)))

# 打乱顺序0.05的情况下，加噪声效果比不加噪声更好


# 未打乱噪声随机（0.1-1）,数据集3，test2
# 原始mae——0.06398691535536155,rmse——0.07855878847070456
# 加噪声mae——0.13004797915751126,rmse——0.16808131369226648
# 加噪声emmae——0.11916431993200571,rmse——0.14929839634534758

# 打乱噪声随机（0.1-1）,数据集3，test2
# 原始mae——0.09107393632834548,rmse——0.10959390178030352
# 加噪声mae——0.13048279369331628,rmse——0.1653084524304821
# 加噪声emmae——0.12326065740853098,rmse——0.15235145098953007


# 均值，由于打乱数据对结果影响太大，所以用原始数据
# 0.01-0.1
# 原始mae——0.06398691535536155,rmse——0.07855878847070456
# 加噪声mae——0.13395967243111928,rmse——0.17060378091860126
# 加噪声emmae——0.1258311504433895,rmse——0.15751751579820564
# 0.01-0.2
# 原始mae——0.06398691535536155,rmse——0.07855878847070456
# 加噪声mae——0.13260469558558013,rmse——0.16778383312120668
# 加噪声emmae——0.12907814072779863,rmse——0.1600540373054632
# 噪声固定在0.05,20
# 原始mae——0.06398691535536156,rmse——0.07855878847070458
# 加噪声mae——0.13567339529805672,rmse——0.16890069477719394
# 加噪声emmae——0.12480172315488676,rmse——0.15794343525939264
# 50次实验
# 原始mae——0.06398691535536155,rmse——0.07855878847070456
# 加噪声mae——0.1414808052082463,rmse——0.17625665207504101
# 加噪声emmae——0.1251652594323094,rmse——0.15879773490499158
# 100次：
# 原始mae——0.06398691535536155,rmse——0.07855878847070456
# 加噪声mae——0.13123149455190483,rmse——0.1679473923317613
# 加噪声emmae——0.12300862934509946,rmse——0.15586986365612743