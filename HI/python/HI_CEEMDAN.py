import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import svm
from sklearn.linear_model import LinearRegression,Ridge,Lasso
import xgboost
import random
from sklearn.neural_network import MLPRegressor
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from PyEMD import CEEMDAN
import statistics

rowdata=pd.read_csv('rowdata.csv')
random_data=rowdata
# random_data=rowdata.sample(frac=1).reset_index(drop=True).iloc[:,1:]

feature1=np.array(random_data['HI10_100'])
feature2=np.array(random_data['HI100_150'])
label=np.array(random_data['Cycle_life'])


# 归一化
from sklearn.preprocessing import MinMaxScaler
def normalization(data):
    scaler = MinMaxScaler()
    data=scaler.fit_transform(np.array(data).reshape(-1,1))  #
    return data


feature1_scale=normalization(feature1)
feature2_scale=normalization(feature2)

# 对特征和数据进行取对数
import math

labellog=[]
for i in label:
    labellog.append(math.log(i,10))
    
labellog=np.array(labellog).reshape(-1,1)
feature1_scale=feature1_scale[:]
feature2_scale=feature2_scale[:]
feature_scale=np.concatenate([feature1_scale, feature2_scale],axis=1)

#Spearman系数


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


X_train,X_test=feature2_scale[:58,:],feature2_scale[58:84,:]
y_train,y_test=labellog[:58],labellog[58:84]


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
#加噪声

def ceemdan(X_train,X_test):
    X_data = np.vstack((X_train, X_test))
    IImfs = []
    data=X_data.ravel()
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
    return X_train,X_test

def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mean_squared_error(y_test, y_predict))
    return mae, rmse

def get_result(X_train,y_train,X_test,y_test):
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
    return mae,rmse

# X_train=add_noise(X_train,0.05)
# X_train,X_test=ceemdan(X_train,X_test)

maelist=[]
rmselist=[]
maelistnoise=[]
rmselistnoise=[]
maelistnoiseemd=[]
rmselistnoiseemd=[]
for i in range(100):
    mae,rmse=get_result(X_train, y_train, X_test, y_test)
    maelist.append(mae)
    rmselist.append(rmse)
    std=random.uniform(0.01, 0.1)
    X_train = add_noise(X_train, std)
    maenoise, rmsenoise = get_result(X_train, y_train, X_test, y_test)
    maelistnoise.append(maenoise)
    rmselistnoise.append(rmsenoise)
    X_train, X_test = ceemdan(X_train, X_test)
    maenoiseemd, rmsenoiseemd = get_result(X_train, y_train, X_test, y_test)
    maelistnoiseemd.append(maenoiseemd)
    rmselistnoiseemd.append(rmsenoiseemd)


print('原始mae——{},rmse——{}'.format(statistics.median(maelist),statistics.median(rmselist)))
print('加噪声mae——{},rmse——{}'.format(statistics.median(maelistnoise),statistics.median(rmselistnoise)))
print('加噪声emmae——{},rmse——{}'.format(statistics.median(maelistnoiseemd),statistics.median(rmselistnoiseemd)))

# 打乱顺序0.05的情况下，加噪声效果比不加噪声更好

# 未打乱0.01
# 原始mae——0.09321287883984354,rmse——0.11757049148043622
# 加噪声mae——0.18300221611653977,rmse——0.1967197420166632
# 加噪声emmae——0.10012698757376759,rmse——0.12342201344539722
# 未打乱0.05
# 原始mae——0.07361712070429227,rmse——0.10205014633622844
# 加噪声mae——0.18285943231729127,rmse——0.20275682855821942
# 加噪声emmae——0.07707713728832284,rmse——0.10505525706095352

# 未打乱噪声随机（0.1-1）
# 原始mae——0.093452105029358,rmse——0.12760624829590045
# 加噪声mae——0.19761688529278817,rmse——0.2220455080142951
# 加噪声emmae——0.0940859885912482,rmse——0.12947617659285227
