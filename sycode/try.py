# # import copy
# # import numpy as np
# # import pandas as pd
# # import random
# #
# #
# # for x in range(6):
# #     times = [random.uniform(0.2, 2) for _ in range(3)]
# #     print(times)
# import copy
# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle
# from sklearn.linear_model import OrthogonalMatchingPursuitCV
# from sklearn.model_selection import LeaveOneOut
# import math
# import sys
# import random
# from sklearn.preprocessing import StandardScaler
# from itertools import permutations
#
# from linear_regression_std import tls, ls
# # from libs.stepwise import cv_stepwise
#
# def rmse(y_true, y_pred):
#     return np.sqrt(sum(np.square(y_true - y_pred)) / len(y_true))
#
# def get_std(a):  # 求一列数据的标准差
#     mean_value = 0  # 设置均值为零
#     variance = np.sum((a-mean_value) ** 2)
#     _std=np.sqrt(variance)
#     return _std
#
#
# items=['F2','F3','F5','F6','F9']
#
# #tls和em算法结合
# def add_em(X, Y,flag,x_test,Y_test,W0, b0):
#     X_test=x_test
#     Y_test=Y_test
#     X_train=X
#     rmse_train = []
#     rmse_test = []
#     # 使用随机初始值
#     # W=np.random.RandomState().uniform(0, 10, (5, 1))
#     # b = np.random.RandomState().uniform(0, 10, (1, 1))
#
#     # 使用固定初始值
#     W = W0
#     b = b0
#     i = 0
#     # dis=1111
#     # print('w:',W)
#     # print('b',b)
#
#     while i<20:# dis>1e-10
#
#         # print(i)
#         xita=[]
#         X_dataframe = copy.deepcopy(X)
#         Y_predict=np.dot(X[..., 0:5], W) + b
#         xita=Y-Y_predict
#
#         _std = []  # 计算三个电池批次的标准差
#         _std.append(get_std(xita[:N_train[0]]))
#         _std.append(get_std(xita[N_train[0]:N_train[1] + N_train[0]]))
#         _std.append(get_std(xita[N_train[1] + N_train[0]:]))
#
#
#         lamuda=[]
#         lamuda.append( 1/_std[0])
#         lamuda.append( 1/_std[1])
#         lamuda.append( 1/_std[2])
#         # #将标准差进行标准化
#         for c in range(len(lamuda)):
#             lamuda[c] = lamuda[c] / (np.sum(lamuda)*0.4)
#
#         #对数据集加权
#         X_dataframe[:N_train[0], 0:6]=X_dataframe[:N_train[0], 0:6]* lamuda[0]
#         X_dataframe[N_train[0]:N_train[1]+N_train[0], 0:6] = X_dataframe[N_train[0]:N_train[1]+N_train[0], 0:6]*lamuda[1]
#         X_dataframe[N_train[1]+N_train[0]:N_train[1]+N_train[0]+N_train[2], 0:6] = X_dataframe[N_train[1]+N_train[0]:N_train[1]+N_train[0]+N_train[2], 0:6] * lamuda[2]
#
#         #加权数据几求模型系数
#         x_temp=np.random.permutation(X_dataframe)
#         X_ = x_temp[:,0:5]
#         Y_ = x_temp[:,5].reshape(-1,1)
#         if flag=='tls':
#             W_em, b_em = tls(X_, Y_)
#         if flag=='ls':
#             W_em, b_em = ls(X_, Y_)
#
#         # dis=np.linalg.norm(W_em-W)
#         W, b = W_em, b_em
#
#         #先查看迭代过程中的rmse
#         mmm=rmse(Y, np.dot(X[:, 0:5], W) + b)
#         rmse_train.append(rmse(Y, np.dot(X[:, 0:5], W) + b))
#         rmse_test.append(rmse(Y_test, np.dot(X_test, W) + b))
#         # xxx=rmse(Y_test, y_pred_tls_em)
#         # print("rmse is :",s)
#         i += 1
#
#
#     # print('rmse_train',rmse_train)
#     # plt.plot(rmse_train)
#     # plt.plot(rmse_test)
#     # plt.legend(['train', 'test'])  #
#     # plt.xlabel('loop')
#     # plt.ylabel('RMSE+{}'.format(flag))
#     # plt.show()
#
#     return W,b
#
#
# # 加载数据
# data_all = pd.read_csv('dataset.csv')
# data = data_all[['F2', 'F3', 'F5', 'F6', 'F9','cyclelife']]  # 注意特征与feature_remain保持一致!!!
# _class = [0] * 41 + [1] * 43 + [2] * 40
# data['class']=_class
# _xita=[0]*124
# data['xita']=_xita
# # print(data)
#
# # 数据集划分
# # data1 = data.iloc[:41, ]  # DataFrame切片
# # data2 = data.iloc[41:84, ]
# # data3 = data.iloc[84:, ]
#
# # 选择数据集
# data_x = copy.deepcopy(data.values)
#
# N_train = []
# N_train.append(round(41 * 0.9))  # 训练集比例
# N_train.append(round(43 * 0.9))
# N_train.append(round(40 * 0.9))
#
# # n = 20  # 最大噪声水平：times=19*0.05，noise_Y = times * standard_Y * np.random.randn(Y_train.shape[0], 1)
# # s = 100 # 分割数据的次数（对数据进行随机排序的次数）
# # m = 50  # 对于每次分割得到的训练集，生成m次噪声
# n = 20    # 最大噪声水平：times=19*0.05，noise_Y = times * standard_Y * np.random.randn(Y_train.shape[0], 1)
# s = 10   # 分割数据的次数（对数据进行随机排序的次数）
# m = 10   # 对于每次分割得到的训练集，生成m次噪声
# w = 6  #5轮噪声比例
#
#
#
# med_tls_rmse = []
# med_ls_rmse = []
# med_tls_em_rmse = []
# med_ls_em_rmse = []
#
# for j in range(n):  # 调整噪声大小
#     np.random.seed(j)
#     print("noise_level:", j)
#     tls_rmse = []
#     ls_rmse = []
#     tls_em_rmse = []
#     ls_em_rmse = []
#     copy_data = data_x
#
#     for x in range(w):
#         # print("不同噪声比例：")
#
#         times_list = [[1, 0.7, 0.01], [0.8, 0.09, 0.00035], [1, 0.18, 0.065], [0.9, 0.4, 0.45], [0.8, 0.9, 0.0452],
#                       [0.93, 0.06, 0.0731]]
#         times = copy.deepcopy(times_list[x])
#         # times = [random.uniform(0, 1) for _ in range(3)]
#         # times[times.index(min(times))] = times[times.index(min(times))] * 0.05
#         # times[times.index(max(times))] = times[times.index(max(times))] * 1
#
#         # 随机打乱数组的顺序
#         # random.shuffle(times)
#         # print(times)
#         times[0] = (times[0] * j * 0.05)
#         times[1] = (times[1] * j * 0.05)
#         times[2] = (times[2] * j * 0.05)
#         # times = []
#         # times.append( 0.1* j * 0.05)
#         # times.append(0.45 * j * 0.05)
#         # times.append(2 * j * 0.05)
#         # times=[]
#         # times.append(random.randint(0, 1) * j )
#         # times.append(random.randint(0, 1) * j )
#         # times.append(random.randint(0, 1) * j )
#
#         # print('j:',j)
#         # print('times[]=:', times)
#         print(times)
#
#         for p in range(s):  # 分割数据
#             # print("不同训练集分割：", p)
#             # 划分训练集与测试集
#             np.random.seed(p)
#             # random_datax = copy_data.reindex(np.random.permutation(copy_data.index))  # 随机排序
#             random_datax = copy_data
#             # print('划分之前的矩阵应该不变的：',random_datax)
#             # 按照每个电池批次进行划分之后再合并
#
#             X_data_1 = np.random.permutation(random_datax[:41, :])
#             X_data_2 = np.random.permutation(random_datax[41:84, :])
#             X_data_3 = np.random.permutation(random_datax[84:, :])
#             # X_data_1 = random_datax[:41, :]
#             # X_data_2 = random_datax[41:84, :]
#             # X_data_3 = random_datax[84:, :]
#             X_train1 = X_data_1[:N_train[0], :]
#             X_test1 = X_data_1[N_train[0]:, :]
#             X_train2 = X_data_2[:N_train[1], :]
#             X_test2 = X_data_2[N_train[1]:, :]
#             X_train3 = X_data_3[:N_train[2], :]
#             X_test3 = X_data_3[N_train[2]:, :]
#
#             data_train_random = np.concatenate((np.concatenate((X_train1, X_train2), axis=0), X_train3), axis=0)
#             X_test_random = np.concatenate((np.concatenate((X_test1, X_test2), axis=0), X_test3), axis=0)
#             # print('data_train_random:',data_train_random)
#
#             data_train_random[..., 5] = np.log10(data_train_random[..., 5])  # dataframe
#             X_test_random[..., 5] = np.log10(X_test_random[..., 5])  # dataframe
#             X_test_random = np.random.permutation(X_test_random)
#             Y_test_random = X_test_random[..., 5].reshape(-1, 1)
#
#             for k in range(m):  # 生成m次噪声
#                 # print("不同噪声矩阵：")
#
#                 X_train = copy.deepcopy(data_train_random)
#                 # Y_train = copy.deepcopy(data_train_random.iloc)
#                 X_test_random = X_test_random[..., 0:5]
#                 X_test = copy.deepcopy(X_test_random)  # dataframe
#                 Y_test = copy.deepcopy(Y_test_random)
#                 standard_X = np.std(X_train, axis=0)  # .reshape(-1, 1)
#                 np.random.seed(k)
#                 noise_X0 = np.random.normal(loc=0, scale=times[0], size=(N_train[0], 6))
#                 noise_X1 = np.random.normal(loc=0, scale=times[1], size=(N_train[1], 6))
#                 noise_X2 = np.random.normal(loc=0, scale=times[2], size=(N_train[2], 6))
#
#                 noise_X = np.concatenate((np.concatenate((noise_X0, noise_X1), axis=0), noise_X2), axis=0)
#                 # print('noise_X is :',noise_X[0])
#                 # print('noise_X is :',noise_X[50])
#
#                 X_train_noise = copy.deepcopy(X_train)
#                 Y_train = X_train_noise[..., 5].reshape(-1, 1)
#
#                 for index in range(len(X_train_noise)):
#                     for i in range(6):
#                         noise_X[:, i] *= (standard_X[i])  # 根据每个特征的标准差生成噪声
#                         X_train_noise[:, i] += noise_X[:, i]
#
#                 # 转换数据类型（前面使用DataFrame是因为之前要进行特征选择）
#                 x_train = X_train_noise
#                 Y_train_noise = np.array(X_train_noise[:, 5]).reshape(-1, 1)
#                 x_test = X_test
#
#                 # # 总体最小二乘
#                 # W_tls, b_tls, = tls(x_train[:,0:5], Y_train_noise)#x:array,y:array,(-1,1)
#                 # W_tls_em,b_tls_em=add_em(x_train, Y_train_noise,'tls',x_test,Y_test)#x:array,y:array,(-1,1)
#                 # y_pred_tls = np.dot(x_test, W_tls) + b_tls
#                 # y_pred_tls_em = np.dot(x_test, W_tls_em) + b_tls_em
#                 # tls_rmse.append(rmse(Y_test, y_pred_tls))
#                 # tls_em_rmse.append(rmse(Y_test, y_pred_tls_em))
#                 # print('tls_em rmse and tls rmse is:',rmse(Y_test, y_pred_tls_em),rmse(Y_test, y_pred_tls))
#
#                 # print("x_train",x_train[0])
#                 # 最小二乘
#                 W_ls, b_ls, = ls(x_train[:, 0:5], Y_train_noise)
#                 W_ls_em, b_ls_em = add_em(x_train, Y_train_noise, 'ls', x_test, Y_test, W_ls, b_ls)
#                 y_pred_ls = np.dot(x_test, W_ls) + b_ls
#                 y_pred_ls_em = np.dot(x_test, W_ls_em) + b_ls_em
#                 # print('ls_em rmse and ls rmse is:', rmse(Y_test, y_pred_ls_em), rmse(Y_test, y_pred_ls))
#                 ls_rmse.append(rmse(Y_test, y_pred_ls))
#                 ls_em_rmse.append(rmse(Y_test, y_pred_ls_em))
#             # print('med_ls_rmse:', ls_rmse)
#             # print('med_ls_em_rmse:', ls_em_rmse)
#     # med_tls_rmse.append(np.median(tls_rmse))
#     med_ls_em_rmse.append(np.median(ls_em_rmse))
#     med_ls_rmse.append(np.median(ls_rmse))
#     med_tls_em_rmse.append(np.median(tls_em_rmse))
#     med_tls_rmse.append(np.median(tls_rmse))
#
#     # 画图tls
#     # plt.plot(med_tls_em_rmse)
#     # plt.plot(med_tls_rmse)
#     # plt.legend([ 'TLS_EM','TLS']) #
#     # plt.xlabel('Noise Level')
#     # plt.ylabel('RMSE')
#     # plt.show()
#
# print('med_ls_em_rmse:', med_ls_em_rmse)
# print('med_ls_rmse:', med_ls_rmse)
# # 画图ls
# # plt.xlim(0,5)
# # x = np.linspace(0, 5, 20)
# x_plt=np.arange(0, 1, 0.05)
# plt.plot(x_plt,med_ls_em_rmse,)
# plt.plot(x_plt,med_ls_rmse)
# plt.legend(['LS_EM', 'LS'])  #
# plt.xlabel('Noise')
# plt.ylabel('RMSE')
# plt.xticks(x_plt)
# plt.locator_params(axis='x', nbins=10)  # 在横坐标上显示6个刻度
# plt.title("data split:{},noise generation:{}".format(s, m))
# plt.show()
#
#
#
# # 画图tls和ls
# # plt.plot(med_tls_rmse)
# # plt.plot(med_ls_rmse)
# # plt.legend(['TLS', 'LS'])  #
# # plt.xlabel('Noise level {}'.format(n*s*m))
# # plt.ylabel('RMSE')
# # plt.show()
#
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
x_plt = np.arange(0.0, 0.49, 0.05)
# x_plt=[0.25,0.3]
print(x_plt)
med_tls_em_rmse=[0.06597122594434598, 0.06595869540652594, 0.06595928463241982, 0.06595233398528197, 0.06596389470965375, 0.0659725983951577, 0.06598380846934598, 0.06601839462387374, 0.06604532379864854, 0.0660538011777303, 0.06606026080509977, 0.06607853231643734]
med_tls_rmse=[0.06640092020031704, 0.06666796239115012, 0.06703738606209829, 0.06730317129306401, 0.06764227766846137, 0.06793855404233806, 0.06839757530534957, 0.06886468569102114, 0.06934443872136833, 0.0698666896608946, 0.07051353280986969, 0.07104463893042127]
med_ls_em_rmse=[0.06666100092954705, 0.06667085377931327, 0.06667085377931327, 0.06666375631791008, 0.06666375631791008, 0.06665343881344782, 0.06665586252484705, 0.06665147975674461, 0.06665094020185715, 0.06665094020185715, 0.06665094020185715, 0.06664259640631523]
med_ls_rmse=[0.0712407250712197, 0.0728008290089924, 0.07439982141404627, 0.07588286993634916, 0.07761466968365568, 0.07933843616370924, 0.08098361732296802, 0.0828353375695461, 0.08459365661626375, 0.08638004518933937, 0.08813518636974149, 0.0899841326076476]

med_tls_em_rmse=[0.06672196870654079, 0.06687546381447823, 0.06666002969564028, 0.06646613447329978, 0.06636494000500051, 0.066292514173211, 0.06625293260562867, 0.06621687866966751, 0.06619736494769948, 0.06617235025635934]
med_tls_rmse=[0.0646047304903378, 0.06474368341673538, 0.06491119073118226, 0.06518748739302188, 0.06534872707828393, 0.06544719476969557, 0.06554390745440293, 0.06562003094946164, 0.0657199241272719, 0.06579637414595406]
med_ls_em_rmse=[0.06645331459065781, 0.06634625988337514, 0.0663250312532393, 0.06631118193086702, 0.06634524285045727, 0.06637121620183165, 0.06642789317610683, 0.06647716737135084, 0.06650761207976048, 0.06651808855877195]
med_ls_rmse=[0.0627574474750485, 0.06258437318996009, 0.06277277016883179, 0.06277812223107497, 0.06284289091908088, 0.06321729184597309, 0.0638734271996455, 0.06437607297736446, 0.0650989241246956, 0.06599784307105011]
plt.plot(x_plt, med_tls_em_rmse, )
plt.plot(x_plt, med_tls_rmse)
plt.plot(x_plt, med_ls_em_rmse, )
plt.plot(x_plt, med_ls_rmse)
plt.legend(['TLS_EM', 'TLS','LS_EM', 'LS'])  #
plt.xlabel('Noise')
plt.ylabel('RMSE')
plt.xticks(x_plt)
plt.locator_params(axis='x', nbins=10)  # 在横坐标上显示6个刻度
plt.title("data split:50,noise generation:20")
plt.show()