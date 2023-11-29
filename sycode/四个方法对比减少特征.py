# import copy
# import numpy as np
# import pandas as pd
# import random
#
#
# for x in range(6):
#     times = [random.uniform(0.2, 2) for _ in range(3)]
#     print(times)
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
from sklearn.preprocessing import StandardScaler
from itertools import permutations

from linear_regression_std import tls, ls


# from libs.stepwise import cv_stepwise

def rmse(y_true, y_pred):
    return np.sqrt(sum(np.square(y_true - y_pred)) / len(y_true))


def get_std(a):  # 求一列数据的标准差
    mean_value = 0  # 设置均值为零
    variance = np.sum((a - mean_value) ** 2)
    _std = np.sqrt(variance)
    return _std


items = ['F2', 'F3', 'F5']


# tls和em算法结合
def add_em(X, Y, flag, x_test, Y_test, W0, b0):
    X_test = x_test
    Y_test = Y_test
    X_train = X
    rmse_train = []
    rmse_test = []
    # 使用随机初始值
    # W=np.random.RandomState().uniform(0, 10, (5, 1))
    # b = np.random.RandomState().uniform(0, 10, (1, 1))

    # 使用固定初始值
    W = W0
    b = b0
    i = 0
    # dis=1111
    # print('w:',W)
    # print('b',b)

    while i < 20:  # dis>1e-10

        # print(i)
        xita = []
        X_dataframe = copy.deepcopy(X)
        Y_predict = np.dot(X[..., 0:3], W) + b
        xita = Y - Y_predict

        _std = []  # 计算三个电池批次的标准差
        _std.append(get_std(xita[:N_train[0]]))
        _std.append(get_std(xita[N_train[0]:N_train[1] + N_train[0]]))
        _std.append(get_std(xita[N_train[1] + N_train[0]:]))

        lamuda = []
        lamuda.append(1 / _std[0])
        lamuda.append(1 / _std[1])
        lamuda.append(1 / _std[2])
        # #将标准差进行标准化
        if flag=='tls':
            for c in range(len(lamuda)):
                lamuda[c] = lamuda[c] / (np.sum(lamuda))
        elif flag=='ls':
            for c in range(len(lamuda)):
                lamuda[c] = lamuda[c] / (np.sum(lamuda)*0.35 )

        # 对数据集加权
        X_dataframe[:N_train[0], 0:4] = X_dataframe[:N_train[0], 0:4] * lamuda[0]
        X_dataframe[N_train[0]:N_train[1] + N_train[0], 0:4] = X_dataframe[N_train[0]:N_train[1] + N_train[0], 0:4] * \
                                                               lamuda[1]
        X_dataframe[N_train[1] + N_train[0]:N_train[1] + N_train[0] + N_train[2], 0:4] = X_dataframe[
                                                                                         N_train[1] + N_train[0]:
                                                                                         N_train[1] + N_train[0] +
                                                                                         N_train[2], 0:4] * lamuda[2]

        # 加权数据几求模型系数
        x_temp = np.random.permutation(X_dataframe)
        X_ = x_temp[:, 0:3]
        Y_ = x_temp[:, 3].reshape(-1, 1)
        if flag == 'tls':
            W_em, b_em = tls(X_, Y_)
        if flag == 'ls':
            W_em, b_em = ls(X_, Y_)

        # dis=np.linalg.norm(W_em-W)
        W, b = W_em, b_em

        # 先查看迭代过程中的rmse
        mmm = rmse(Y, np.dot(X[:, 0:3], W) + b)
        rmse_train.append(rmse(Y, np.dot(X[:, 0:3], W) + b))
        rmse_test.append(rmse(Y_test, np.dot(X_test, W) + b))
        # xxx=rmse(Y_test, y_pred_tls_em)
        # print("rmse is :",s)
        i += 1

    # print('rmse_train',rmse_train)
    # plt.plot(rmse_train)
    # plt.plot(rmse_test)
    # plt.legend(['train', 'test'])  #
    # plt.xlabel('loop')
    # plt.ylabel('RMSE+{}'.format(flag))
    # plt.show()

    return W, b


# 加载数据
data_all = pd.read_csv('dataset.csv')
data = data_all[['F2', 'F3', 'F5', 'cyclelife']]  # 注意特征与feature_remain保持一致!!!
_class = [0] * 41 + [1] * 43 + [2] * 40
data['class'] = _class
_xita = [0] * 124
data['xita'] = _xita
# print(data)

# 数据集划分
# data1 = data.iloc[:41, ]  # DataFrame切片
# data2 = data.iloc[41:84, ]
# data3 = data.iloc[84:, ]

# 选择数据集
data_x = copy.deepcopy(data.values)

N_train = []
N_train.append(round(41 * 0.9))  # 训练集比例
N_train.append(round(43 * 0.9))
N_train.append(round(40 * 0.9))

# n = 20  # 最大噪声水平：times=19*0.05，noise_Y = times * standard_Y * np.random.randn(Y_train.shape[0], 1)
# s = 100 # 分割数据的次数（对数据进行随机排序的次数）
# m = 50  # 对于每次分割得到的训练集，生成m次噪声
n = 20  # 最大噪声水平：times=19*0.05，noise_Y = times * standard_Y * np.random.randn(Y_train.shape[0], 1)
s = 50  # 分割数据的次数（对数据进行随机排序的次数）
m = 20 # 对于每次分割得到的训练集，生成m次噪声
w = 4  # 5轮噪声比例
# times_list = [[1, 0.1, 0.02], [0.98, 0.1, 0.06], [0.96, 0.18, 0.05],[0.97, 0.07, 0.06],[0.95, 0.2, 0.03],[1, 0.07, 0.05],[0.99, 0.2, 0.03],[0.97, 0.04, 0.06] ]

times_list = [[1, 0.1, 0.02],[0.97, 0.07, 0.05],[0.99, 0.04, 0.03],[1,0.06,0.08]]
#

# times_list = [[0.98, 0.1, 0.06]]
n_list=np.arange(0,  10, 1)
# n_list=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
for x in range(w):

    # times = [random.uniform(0, 1) for _ in range(3)]
    # times[times.index(min(times))] = times[times.index(min(times))] * 0.05
    # times[times.index(max(times))] = times[times.index(max(times))] * 1

    # 随机打乱数组的顺序
    # random.shuffle(times)
    # print(times)
    med_tls_rmse = []
    med_ls_rmse = []
    med_tls_em_rmse = []
    med_ls_em_rmse = []
    tls_rmse = []
    ls_rmse = []
    tls_em_rmse = []
    ls_em_rmse = []

    for j in n_list:  # 调整噪声大小

        np.random.seed(j)
        times = copy.deepcopy(times_list[x])

        copy_data = data_x

        times[0] = (times[0] * j * 0.05)
        times[1] = (times[1] * j * 0.05)
        times[2] = (times[2] * j * 0.05)

        print(times)
        print("noise_level:", j)
        for p in range(s):  # 分割数据
            # print("不同训练集分割：", p)
            # 划分训练集与测试集
            np.random.seed(p)
            # random_datax = copy_data.reindex(np.random.permutation(copy_data.index))  # 随机排序
            random_datax = copy_data
            # print('划分之前的矩阵应该不变的：',random_datax)
            # 按照每个电池批次进行划分之后再合并

            X_data_1 = np.random.permutation(random_datax[:41, :])
            X_data_2 = np.random.permutation(random_datax[41:84, :])
            X_data_3 = np.random.permutation(random_datax[84:, :])
            # X_data_1 = random_datax[:41, :]
            # X_data_2 = random_datax[41:84, :]
            # X_data_3 = random_datax[84:, :]
            X_train1 = X_data_1[:N_train[0], :]
            X_test1 = X_data_1[N_train[0]:, :]
            X_train2 = X_data_2[:N_train[1], :]
            X_test2 = X_data_2[N_train[1]:, :]
            X_train3 = X_data_3[:N_train[2], :]
            X_test3 = X_data_3[N_train[2]:, :]

            data_train_random = np.concatenate((np.concatenate((X_train1, X_train2), axis=0), X_train3), axis=0)
            X_test_random = np.concatenate((np.concatenate((X_test1, X_test2), axis=0), X_test3), axis=0)
            # print('data_train_random:',data_train_random)

            data_train_random[..., 3] = np.log10(data_train_random[..., 3])  # dataframe
            X_test_random[..., 3] = np.log10(X_test_random[..., 3])  # dataframe
            X_test_random = np.random.permutation(X_test_random)
            Y_test_random = X_test_random[..., 3].reshape(-1, 1)

            for k in range(m):  # 生成m次噪声
                # print("不同噪声矩阵：")

                X_train = copy.deepcopy(data_train_random)
                # Y_train = copy.deepcopy(data_train_random.iloc)
                X_test_random = X_test_random[..., 0:3]
                X_test = copy.deepcopy(X_test_random)  # dataframe
                Y_test = copy.deepcopy(Y_test_random)
                standard_X = np.std(X_train, axis=0)  # .reshape(-1, 1)
                np.random.seed(k)
                noise_X0 = np.random.normal(loc=0, scale=times[0], size=(N_train[0], 4))
                noise_X1 = np.random.normal(loc=0, scale=times[1], size=(N_train[1], 4))
                noise_X2 = np.random.normal(loc=0, scale=times[2], size=(N_train[2], 4))

                noise_X = np.concatenate((np.concatenate((noise_X0, noise_X1), axis=0), noise_X2), axis=0)
                # print('noise_X is :',noise_X[0])
                # print('noise_X is :',noise_X[50])

                X_train_noise = copy.deepcopy(X_train)
                Y_train = X_train_noise[..., 3].reshape(-1, 1)

                for index in range(len(X_train_noise)):
                    for i in range(4):
                        noise_X[:, i] *= (standard_X[i])  # 根据每个特征的标准差生成噪声
                        X_train_noise[:, i] += noise_X[:, i]

                # 转换数据类型（前面使用DataFrame是因为之前要进行特征选择）
                x_train = X_train_noise
                Y_train_noise = np.array(X_train_noise[:, 3]).reshape(-1, 1)
                x_test = X_test

                # # 总体最小二乘
                W_tls, b_tls, = tls(x_train[:,0:3], Y_train_noise)#x:array,y:array,(-1,1)
                W_tls_em,b_tls_em=add_em(x_train, Y_train_noise,'tls',x_test,Y_test, W_tls, b_tls)#x:array,y:array,(-1,1)
                y_pred_tls = np.dot(x_test, W_tls) + b_tls
                y_pred_tls_em = np.dot(x_test, W_tls_em) + b_tls_em
                tls_rmse.append(rmse(Y_test, y_pred_tls))
                tls_em_rmse.append(rmse(Y_test, y_pred_tls_em))

                # 最小二乘
                W_ls, b_ls, = ls(x_train[:, 0:3], Y_train_noise)
                W_ls_em, b_ls_em = add_em(x_train, Y_train_noise, 'ls', x_test, Y_test, W_ls, b_ls)
                y_pred_ls = np.dot(x_test, W_ls) + b_ls
                y_pred_ls_em = np.dot(x_test, W_ls_em) + b_ls_em
                # print('ls_em rmse and ls rmse is:', rmse(Y_test, y_pred_ls_em), rmse(Y_test, y_pred_ls))
                ls_rmse.append(rmse(Y_test, y_pred_ls))
                ls_em_rmse.append(rmse(Y_test, y_pred_ls_em))


            # print('med_ls_rmse:', ls_rmse)
            # print('med_ls_em_rmse:', ls_em_rmse)
        # med_tls_rmse.append(np.median(tls_rmse))
        med_ls_em_rmse.append(np.median(ls_em_rmse))
        med_ls_rmse.append(np.median(ls_rmse))
        med_tls_em_rmse.append(np.median(tls_em_rmse))
        med_tls_rmse.append(np.median(tls_rmse))
        print('jjjjj')

    # 画图tls
    # plt.plot(med_tls_em_rmse)
    # plt.plot(med_tls_rmse)
    # plt.legend([ 'TLS_EM','TLS']) #
    # plt.xlabel('Noise Level')
    # plt.ylabel('RMSE')
    # plt.show()

    print('med_tls_em_rmse:', med_tls_em_rmse)
    print('med_tls_rmse:', med_tls_rmse)
    print('med_ls_em_rmse:', med_ls_em_rmse)
    print('med_ls_rmse:', med_ls_rmse)
    # 画图ls
    # plt.xlim(0,5)
    # x = np.linspace(0, 5, 20)
    plt.figure(figsize=(7, 5))
    x_plt = np.arange(0.0, 0.52, 0.05)
    # x_plt=[0.25,0.3]
    print(x_plt)
    plt.plot(x_plt, med_tls_em_rmse, )
    plt.plot(x_plt, med_tls_rmse)
    plt.plot(x_plt, med_ls_em_rmse, )
    plt.plot(x_plt, med_ls_rmse)
    plt.legend(['TLS_EM', 'TLS','LS_EM', 'LS'])  #
    plt.xlabel('Noise')
    plt.ylabel('RMSE')
    plt.xticks(x_plt)
    plt.locator_params(axis='x', nbins=10)  # 在横坐标上显示6个刻度
    plt.title("data split:{},noise generation:{}".format(s, m))
    plt.show()

# 画图tls和ls
# plt.plot(med_tls_rmse)
# plt.plot(med_ls_rmse)
# plt.legend(['TLS', 'LS'])  #
# plt.xlabel('Noise level {}'.format(n*s*m))
# plt.ylabel('RMSE')
# plt.show()

