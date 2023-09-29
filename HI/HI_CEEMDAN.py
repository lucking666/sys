import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
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
import numpy as np
import pandas as pd
import scipy.stats as stats
def calculate_spearman_correlation(X, Y):
    r,p=stats.spearmanr(X, Y)
    return r,p

r,p=calculate_spearman_correlation(feature1,label)
print("相关系数：",r)

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

X_train=add_noise(X_train,0.05)


X_data=np.vstack((X_train, X_test))

from PyEMD import CEEMDAN

# 生成res的分解
def ceemdan_decompose_res(data):
    ceemdan = CEEMDAN()
    ceemdan.ceemdan(data)
    imfs, res = ceemdan.get_imfs_and_residue()
    plt.figure(figsize=(12,9))
    plt.subplots_adjust(hspace=0.1)
    plt.subplot(imfs.shape[0]+3, 1, 1)
    plt.plot(data,'r')
    for i in range(imfs.shape[0]):
        plt.subplot(imfs.shape[0]+3,1,i+2)
        plt.plot(imfs[i], 'g')
        plt.ylabel("IMF %i" %(i+1))
        plt.locator_params(axis='x', nbins=10)
        # 在函数前必须设置一个全局变量 IImfs=[]
        IImfs.append(imfs[i])
    plt.subplot(imfs.shape[0]+3, 1, imfs.shape[0]+3)
    plt.plot(res,'g')
    return res

# ceemdan分解
IImfs=[]
res=ceemdan_decompose_res(X_data.ravel())


X_train,X_test=np.array(np.transpose(IImfs))[:58,:],np.array(np.transpose(IImfs))[58:84,:]
print(X_train.shape)
print(X_test.shape)

from sklearn.linear_model import LinearRegression,Ridge,Lasso
import xgboost as xgb
from sklearn import svm
from sklearn.neural_network import MLPRegressor


# model1=svm.SVR(probability = True,kernel = 'rbf',c=0.1,max_iter=10)
# model1.fit(X_train,y_train)
# y_pred=model1.predict(X_test)

# lr=LinearRegression().fit(X_train,y_train)
# y_pred=lr.predict(X_test)


xgb = xgb.XGBRegressor()
xgb.fit(X_train, y_train)
y_pred=xgb.predict(X_test)

# gpr = GaussianProcessRegressor()
# gpr.fit(X_train,y_train)
# y_pred=gpr.predict(X_test)

# mlp=MLPRegressor(hidden_layer_sizes = (500,500,500))
# mlp.fit(X_train,y_train)
# y_pred=mlp.predict(X_test)


from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mean_squared_error(y_test, y_predict))
    return mae, rmse

mae,rmse=evaluation(y_test, y_pred)
print('mae——{},rmse——{}'.format(mae,rmse))