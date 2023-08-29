
import copy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.model_selection import LeaveOneOut
import math
import sys
import random
from math import gcd
from sklearn.preprocessing import StandardScaler

from linear_regression_std import tls, ls
# from libs.stepwise import cv_stepwise

def rmse(y_true, y_pred):
    return np.sqrt(sum(np.square(y_true - y_pred)) / len(y_true))

def get_std(a):  # 求一列数据的标准差
    mean_value = 0  # 设置均值为零
    variance = np.sum((a-mean_value) ** 2)/len(a)
    _std=np.sqrt(variance)
    return _std

items=['F2','F3','F5','F6','F9']

#tls和em算法结合
def add_em(X, Y,flag,W0, b0):

    rmse_train = []
    rmse_train.append(1e6)
    temp_rmse=100000
    temp_rmse_dis=1e6
    dis=1e6
    # 使用纯TLS/LS解出来的初始值
    W = W0
    b = b0
    # W=np.random.RandomState().uniform(0, 10, (5, 1))
    # b = np.random.RandomState().uniform(0, 10, (1, 1))

    i = 1

    while i<100:#dis>1e-5

        xita=[]
        X_dataframe = copy.deepcopy(X)
        Y_predict=np.dot(X[..., 0:5], W) + b   #用当前的W,b计算预测值
        xita=Y-Y_predict   #计算预测值和真实值的误差



        _std=[] #计算三个电池批次的标准差
        _std.append(get_std(xita[:N_train[0]]))
        _std.append(get_std(xita[N_train[0]:N_train[1] + N_train[0]]))
        _std.append(get_std(xita[N_train[1] + N_train[0]:]))

        # xita.append((Y[:N_train[0]-(np.dot(X[:N_train[0], 0:5], W) + b)).std())
        # xita.append((Y[N_train[0]:N_train[1]+N_train[0]] - (np.dot(X[N_train[0]:N_train[1]+N_train[0], 0:5], W) + b)).std())
        # xita.append((Y[N_train[1]+N_train[0]:] - (np.dot(X[N_train[1]+N_train[0]:, 0:5], W) + b)).std())
        lamuda=[]  #取标准差的倒数
        lamuda.append( 1/_std[0])
        lamuda.append(1/_std[1])
        lamuda.append(1/_std[2])


        # 将标准差倒数进行标准化
        sum_lamuda=np.sum(lamuda)
        for c in range(len(lamuda)):
            lamuda[c] = lamuda[c] / sum_lamuda

        #对数据集加权
        X_dataframe[:N_train[0], 0:6]=X_dataframe[:N_train[0], 0:6] *lamuda[0]
        X_dataframe[N_train[0]:N_train[1]+N_train[0], 0:6] = X_dataframe[N_train[0]:N_train[1]+N_train[0], 0:6]*lamuda[1]
        X_dataframe[N_train[1]+N_train[0]:N_train[1]+N_train[0]+N_train[2], 0:6] = X_dataframe[N_train[1]+N_train[0]:N_train[1]+N_train[0]+N_train[2], 0:6]*lamuda[2]

        #加权数据集求模型系数
        X_ = X_dataframe[:,0:5]
        Y_ = X_dataframe[:,5].reshape(-1,1)
        if flag=='tls':
            W_em, b_em = tls(X_, Y_)
        if flag=='ls':
            W_em, b_em = ls(X_, Y_)

        # 先查看迭代过程中的rmse
        temp_rmse = rmse(Y, np.dot(X[:, 0:5], W_em) + b_em)
        temp_rmse_dis = rmse_train[-1] - temp_rmse

        if(temp_rmse_dis<-60):
            break
        rmse_train.append(temp_rmse)
        dis=np.linalg.norm(W_em-W)
        W, b = W_em, b_em

        # print("rmse is :",s)
        i += 1

    # print(W,b)
    # print('rmse_train',rmse_train)
    # plt.plot(rmse_train)
    # plt.legend(['train', 'test'])  #
    # plt.xlabel('loop')
    # plt.ylabel('RMSE+{}'.format(flag))
    # plt.show()

    return W,b

def generate_data():
    mu, sigma = 0, 5
    x = np.random.normal(mu, sigma, (124, 5))
    x = np.clip(x, -100, 100)  # 将所有值限制在[-100,100]范围内
    w = np.array([[10.0], [76.0], [1.0], [3.0], [9.0]])
    y = np.dot(x, w) + 1500.0
    a_xy = np.hstack([x, y])
    # print(x, "\ny:", y)
    return x, y, a_xy
q,w,get_data=generate_data()

_getdata=pd.DataFrame(data=get_data,columns=['F2', 'F3', 'F5', 'F6', 'F9','cyclelife'])# 加载数据
data=_getdata
# data_all = pd.read_csv('createdataset.csv')
# data = data_all[['F2', 'F3', 'F5', 'F6', 'F9','cyclelife']]  # 注意特征与feature_remain保持一致!!!

_class = [0] * 41 + [1] * 43 + [2] * 40
data['class']=_class
_xita=[0]*124
data['xita']=_xita
# print(data)


# 选择数据集

data_x = copy.deepcopy(data.values)

N_train = []
N_train.append(round(41 * 0.9))  # 训练集比例
N_train.append(round(43 * 0.9))
N_train.append(round(40 * 0.9))

# n = 20  # 最大噪声水平：times=19*0.05，noise_Y = times * standard_Y * np.random.randn(Y_train.shape[0], 1)
# s = 100 # 分割数据的次数（对数据进行随机排序的次数）
# m = 50  # 对于每次分割得到的训练集，生成m次噪声
n = 20    # 最大噪声水平：times=19*0.05，noise_Y = times * standard_Y * np.random.randn(Y_train.shape[0], 1)
s = 1   # 分割数据的次数（对数据进行随机排序的次数）
m = 1    # 对于每次分割得到的训练集，生成m次噪声
w = 1    #5轮噪声比例

med_tls_rmse = []
med_ls_rmse = []
med_tls_em_rmse = []
med_ls_em_rmse = []
for j in range(n):  # 调整噪声大小
    # np.random.seed(j)
    print("noise_level:",j)
    tls_rmse = []
    ls_rmse = []
    tls_em_rmse = []
    ls_em_rmse = []
    copy_data = data_x
    for x in range(w):
        # print("不同噪声比例：")

        random.seed(x)
        times = [random.uniform(0, 2) for _ in range(3)]
        times[times.index(min(times))] =times[times.index(min(times))]*0.01
        times[times.index(max(times))] =times[times.index(max(times))]*100
        # times=[1,1,1]
        times[0] = (times[0] * j )
        times[1] = (times[1] * j )
        times[2] = (times[2] * j )
        print(times)

        # times = []
        # times.append( 0.1* j * 0.05)
        # times.append(0.45 * j * 0.05)
        # times.append(2 * j * 0.05)
        # times=[]
        # times.append(random.randint(0, 1) * j )
        # times.append(random.randint(0, 1) * j )
        # times.append(random.randint(0, 1) * j )

        # print('j:',j)
        # print('times[]=:', times)

        for p in range(s):  # 分割数据
            # print("不同训练集分割：")
            # 划分训练集与测试集
            np.random.seed(p)
            random_datax = copy_data
            # 按照每个电池批次进行划分之后再合并

            X_data_1 = np.random.permutation(random_datax[:41, :])
            X_data_2 = np.random.permutation(random_datax[41:84, :])
            X_data_3 = np.random.permutation(random_datax[84:, :])
            X_train1 = X_data_1[:N_train[0], :]
            X_test1 = X_data_1[N_train[0]:, :]
            X_train2 = X_data_2[:N_train[1], :]
            X_test2 = X_data_2[N_train[1]:, :]
            X_train3 = X_data_3[:N_train[2], :]
            X_test3 = X_data_3[N_train[2]:, :]


            data_train_random = np.concatenate((np.concatenate((X_train1, X_train2),axis=0), X_train3),axis=0)
            X_test_random = np.concatenate((np.concatenate((X_test1, X_test2),axis=0), X_test3),axis=0)
            # print('data_train_random:',data_train_random)


            data_train_random[...,5] = np.log10(data_train_random[...,5])  # dataframe
            X_test_random[...,5] = np.log10(X_test_random[...,5])  # dataframe
            X_test_random=np.random.permutation(X_test_random)
            Y_test_random = X_test_random[...,5].reshape(-1, 1)


            for k in range(m):  # 生成m次噪声
                # print("不同噪声矩阵：")


                X_train = copy.deepcopy(data_train_random)
                X_test_random = X_test_random[..., 0:5]
                X_test = copy.deepcopy(X_test_random)  # dataframe
                Y_test = copy.deepcopy(Y_test_random)
                standard_X = np.std(X_train, axis=0)  # .reshape(-1, 1)
                np.random.seed(k)
                noise_X0 = np.random.normal(loc=0, scale=times[0], size=(N_train[0], 6))
                noise_X1 = np.random.normal(loc=0, scale=times[1], size=(N_train[1], 6))
                noise_X2 = np.random.normal(loc=0, scale=times[2], size=(N_train[2], 6))

                noise_X = np.concatenate((np.concatenate((noise_X0, noise_X1), axis=0), noise_X2), axis=0)


                X_train_noise = copy.deepcopy(X_train)
                Y_train=X_train_noise[...,5].reshape(-1,1)


                for index in range(len(X_train_noise)):
                    flag = int(X_train_noise[index][6])
                    for i in range(6):
                        noise_X[:, i] *= (standard_X[i])  # 根据每个特征的标准差生成噪声
                        X_train_noise[:, i] += noise_X[:, i]

                # 转换数据类型（前面使用DataFrame是因为之前要进行特征选择）
                x_train = X_train_noise
                Y_train_noise = np.array(X_train_noise[:, 5]).reshape(-1, 1)
                x_test = X_test


                # 总体最小二乘
                W_tls, b_tls, = tls(x_train[:,0:5], Y_train_noise)
                W_tls_em,b_tls_em=add_em(x_train, Y_train_noise,'tls',W_tls, b_tls)
                y_pred_tls = np.dot(x_test, W_tls) + b_tls
                y_pred_tls_em = np.dot(x_test, W_tls_em) + b_tls_em
                tls_rmse.append(rmse(Y_test, y_pred_tls))
                tls_em_rmse.append(rmse(Y_test, y_pred_tls_em))

                # 最小二乘
                W_ls, b_ls, = ls(x_train[:, 0:5], Y_train_noise)
                # print('-------------------------------------------',W_ls,b_ls)
                W_ls_em, b_ls_em = add_em(x_train, Y_train_noise, 'ls', W_ls,b_ls)
                y_pred_ls = np.dot(x_test, W_ls) + b_ls
                y_pred_ls_em = np.dot(x_test, W_ls_em) + b_ls_em
                ls_rmse.append(rmse(Y_test, y_pred_ls))
                ls_em_rmse.append(rmse(Y_test, y_pred_ls_em))


    med_ls_em_rmse.append(np.median(ls_em_rmse))
    med_ls_rmse.append(np.median(ls_rmse))
    med_tls_em_rmse.append(np.median(tls_em_rmse))
    med_tls_rmse.append(np.median(tls_rmse))


print('med_tls_em_rmse:',med_tls_em_rmse)
print('med_tls_rmse:',med_tls_rmse)
# 画图tls
plt.plot(med_tls_em_rmse)
plt.plot(med_tls_rmse)
plt.legend([ 'TLS_EM','TLS']) #
plt.xlabel('Noise Level')
plt.ylabel('RMSE')
plt.show()


#
print('med_ls_em_rmse:',med_ls_em_rmse)
print('med_ls_rmse:',med_ls_rmse)
# # 画图ls
plt.plot(med_ls_em_rmse)
plt.plot(med_ls_rmse)
plt.legend(['LS_EM', 'LS']) #
plt.xlabel('Noise Level')
plt.ylabel('RMSE')
plt.show()

# print('med_tls_rmse:',med_tls_rmse)
# print('med_tls_rmse:',med_ls_rmse)
# # 画图tls和ls
# plt.plot(med_tls_rmse)
# plt.plot(med_ls_rmse)
# plt.legend(['TLS', 'LS'])  #
# plt.xlabel('Noise level {}'.format(n*s*m))
# plt.ylabel('RMSE')
# plt.show()

