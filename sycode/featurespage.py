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


import numpy as np
import matplotlib.gridspec as gridspec
data_all = pd.read_csv('dataset.csv')
data = data_all[['F2', 'F3', 'F5','cyclelife']]  # 注意特征与feature_remain保持一致!!!
_class = [0] * 41 + [1] * 43 + [2] * 40
data['class']=_class


# 获取不同 class 值的颜色映射
colors = {0: 'red', 1: 'green', 2: 'blue'}

# 创建 1 行 3 列的图，确保每个子图是正方形
fig = plt.figure(figsize=(15, 5))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])

# 遍历 'F2', 'F3', 'F5'
for i, feature in enumerate(['F2', 'F3', 'F5']):
    # 在每个子图中绘制散点图
    axs = plt.subplot(gs[i])
    for cls, color in colors.items():
        subset = data[data['class'] == cls]
        axs.scatter(subset.index, subset[feature], label=f'class {cls}', color=color)

    # 设置图标题和标签
    axs.set_xlabel('battery cell', fontsize=12)
    axs.set_ylabel(feature, fontsize=12)
    axs.legend()
fig.text(0.01, 0.93, 'a', fontsize=14, fontweight='bold')
fig.text(0.33, 0.93, 'b', fontsize=14, fontweight='bold')
fig.text(0.68, 0.93, 'c', fontsize=14, fontweight='bold')
# 调整布局
plt.subplots_adjust(top=0.9, hspace=0.6, wspace=0.4, left=0.05, right=0.95)  # 调整顶部留白、子图之间的垂直间距、水平间距，左右留白
# plt.subplots_adjust(top=0.9, hspace=0.4, left=0.05, right=0.95)  # 调整顶部留白、子图之间的垂直间距，左右留白
plt.show()

# med_tls_em_rmse0=[0.10739202428228475, 0.09372489429832595, 0.08894437664393298, 0.08746814354927684, 0.0805287675551736, 0.07631105441331074, 0.07256498038830031, 0.07074661682297248, 0.0663424153361223]
# med_tls_rmse0=[0.24180531239573386, 0.16726337760792784, 0.14268178893446404, 0.12819288298057943, 0.11059570043537237, 0.1040410916615759, 0.09650127572830089, 0.08913648878641837, 0.08037076053834957]
# med_ls_em_rmse0=[0.10334282496788533, 0.0933980035248172, 0.08953837814079055, 0.08873002092683008, 0.08316023899237245, 0.07869102788456843, 0.07476660612033915, 0.07262161465731717, 0.06874818650035505]
# med_ls_rmse0=[0.14613540381067502, 0.12957470364233442, 0.12318252882292714, 0.12122241825884233, 0.11730007306777657, 0.11666292465177139, 0.11516532573677707, 0.11273416264047069, 0.10530821390728343]
#
# med_tls_em_rmse1=[0.10750407384036514, 0.09313456151863742, 0.08847369750602299, 0.08684443193081057, 0.08010939249141089, 0.07594077763422255, 0.07224368780480583, 0.07050003744125236, 0.06590492866487202]
# med_tls_rmse1=[0.23245050554072022, 0.16105460024607712, 0.1375666729559965, 0.12391934545244487, 0.10746350008962349, 0.10136755951274085, 0.09417552861227357, 0.08724217841921192, 0.07880901756525664]
# med_ls_em_rmse1=[0.10361283551172809, 0.09289698543982464, 0.0892996430008718, 0.08852262389668891, 0.08322636921274051, 0.0787096265745818, 0.07458131456294631, 0.0723418787881559, 0.06836187674308145]
# med_ls_rmse1=[0.14338418659904167, 0.12733680019653149, 0.12095816882492791, 0.11908499836119471, 0.11528325844198253, 0.11469120464308288, 0.1132542140899995, 0.11081098686482148, 0.10347172825018021]
#
#
# med_tls_em_rmse2=[0.10606009252358942, 0.09247784965077381, 0.08768997835208166, 0.08588582096294924, 0.07825281574567897, 0.07471169974178306, 0.07102197487362344, 0.06980458642681989, 0.06479573706283975]
# med_tls_rmse2=[0.23790441674417503, 0.16452901346994447, 0.1405355728153337, 0.12668903202665724, 0.10923519843043321, 0.10288789701335276, 0.09554692577326829, 0.08839737875084634, 0.0797739190329706]
# med_ls_em_rmse2=[0.10253267020599204, 0.09209101723041144, 0.0883305704883973, 0.08753395455870115, 0.0811606306929946, 0.07695010488180598, 0.0730116240630933, 0.07121665636984083, 0.06696102396983822]
# med_ls_rmse2=[0.14470570533690194, 0.12852904685951955, 0.12219044781309084, 0.12032952868851193, 0.11640383432470405, 0.11584932959806898, 0.11431644167898984, 0.1119528805113238, 0.10448715538906805]
#
# med_tls_em_rmse3=[0.11095553995399518, 0.09399346358440307, 0.08882996351062686, 0.0870357010999185, 0.08136778791175305, 0.07713542995478626, 0.07304045687017915, 0.07102129765963822, 0.06647776957267915]
# med_tls_rmse3=[0.24192843234212072, 0.16657420938958023, 0.14266661058252983, 0.1280696346388221, 0.11044275948751277, 0.10399661240835632, 0.09642740351923593, 0.08906012393806825, 0.08042434035084253]
# med_ls_em_rmse3=[0.1055317934246299, 0.09360649609990329, 0.08980766476495551, 0.08884505420219392, 0.08487318055300752, 0.08048569019006666, 0.0759578350940012, 0.07341911577993365, 0.06933820745540453]
# med_ls_rmse3=[0.14584921656321012, 0.12969094273230375, 0.12316931884348342, 0.1211960386764619, 0.11729847641419834, 0.11670241876762864, 0.11521570231280788, 0.11279881763800256, 0.10534715966549481]
#
#
# x_plt = train_array = np.arange(0.15, 1.0, 0.1)
#
# fig, axs = plt.subplots(2, 2, figsize=(12, 10))
#
# # 在每个子图中绘制对应的数据
# axs[0, 0].plot(x_plt, med_tls_em_rmse0,label='TLS_EM',marker='o', linestyle='-' )
# axs[0, 0].plot(x_plt, med_tls_rmse0,label='TLS',marker='o', linestyle='-')
# axs[0, 0].plot(x_plt, med_ls_em_rmse0,label='LS_EM',marker='o', linestyle='-' )
# axs[0, 0].plot(x_plt, med_ls_rmse0,label='LS',marker='o', linestyle='-' )
# # axs[0, 0].set_xlabel('Noise')
# # axs[0, 0].set_ylabel('RMSE')
# # axs[0, 0].set_xticks(x_plt)
# axs[0, 0].set_title('[1, 0.1, 0.02]')
# axs[0, 0].locator_params(axis='x', nbins=10)  # 在横坐标上显示6个刻度
# # axs[0, 0].title("data split:{},noise generation:{}".format(s, m))
#
# axs[0, 1].plot(x_plt, med_tls_em_rmse1,label='TLS_EM',marker='o', linestyle='-')
# axs[0, 1].plot(x_plt, med_tls_rmse1, label='TLS',marker='o', linestyle='-')
# axs[0, 1].plot(x_plt, med_ls_em_rmse1, label='LS_EM',marker='o', linestyle='-')
# axs[0, 1].plot(x_plt, med_ls_rmse1, label='LS',marker='o', linestyle='-')
# # axs[0, 1].set_xlabel('Noise')
# # axs[0, 1].set_ylabel('RMSE')
# axs[0, 1].set_title('[0.97, 0.07, 0.05]')
# axs[0, 1].set_xticks(x_plt)
# axs[0, 1].locator_params(axis='x', nbins=10)  # 在横坐标上显示6个刻度
#
#
#
# axs[1, 0].plot(x_plt, med_tls_em_rmse2, label='TLS_EM',marker='o', linestyle='-' )
# axs[1, 0].plot(x_plt, med_tls_rmse2, label='TLS',marker='o', linestyle='-')
# axs[1, 0].plot(x_plt, med_ls_em_rmse2, label='LS_EM',marker='o', linestyle='-' )
# axs[1, 0].plot(x_plt, med_ls_rmse2, label='LS',marker='o', linestyle='-')
# axs[1, 0].set_title('[0.99, 0.04, 0.03]')
# axs[1, 0].set_xticks(x_plt)
# axs[1, 0].locator_params(axis='x', nbins=10)  # 在横坐标上显示6个刻度
#
# axs[1, 1].plot(x_plt, med_tls_em_rmse3, label='TLS_EM',marker='o', linestyle='-' )
# axs[1, 1].plot(x_plt, med_tls_rmse3, label='TLS',marker='o', linestyle='-')
# axs[1, 1].plot(x_plt, med_ls_em_rmse3, label='LS_EM' ,marker='o', linestyle='-')
# axs[1, 1].plot(x_plt, med_ls_rmse3, label='LS',marker='o', linestyle='-')
# # axs[1, 1].set_xlabel('Noise')
# # axs[1, 1].set_ylabel('RMSE')
# axs[1, 1].set_title('[1,0.06,0.08]')
# axs[1, 1].set_xticks(x_plt)
# axs[1, 1].locator_params(axis='x', nbins=10)  # 在横坐标上显示6个刻度
# # 添加标题和标签
# # fig.suptitle('Comparison of RMSE for Four Methods')
# for ax in axs.flat:
#     ax.set(xlabel='train_size', ylabel='RMSE')
#     ax.legend()
#
#
#
#
#
#
#
# #
# # x_plt = np.arange(0.25, 1.01, 0.05)
# #     # x_plt=[0.25,0.3]
# # print(x_plt)
# # plt.plot(x_plt,med_tls_em_rmse, )
# # plt.plot(x_plt,med_tls_rmse)
# # plt.plot(x_plt,med_ls_em_rmse, )
# # plt.plot(x_plt,med_ls_rmse)
# # plt.legend(['TLS_EM', 'TLS','LS_EM', 'LS'])  #
# # plt.xlabel('Noise')
# # plt.ylabel('RMSE')
# #
# # plt.title("data split:50,noise generation:20")
# # plt.show()
#
# #
# # # 绘制折线图
# # plt.figure(figsize=(10, 6))
# # plt.plot(x_plt, med_tls_em_rmse, label='Med TLS EM RMSE', marker='o')
# # plt.plot(x_plt, med_tls_rmse, label='Med TLS RMSE', marker='o')
# # plt.plot(x_plt, med_ls_em_rmse, label='Med LS EM RMSE', marker='o')
# # plt.plot(x_plt, med_ls_rmse, label='Med LS RMSE', marker='o')
# #
# # # 添加标题和标签
# # plt.title('Comparison of RMSE for Four Methods')
# # plt.xlabel('X-axis Label')
# # plt.ylabel('RMSE Values')
# #
# # # 添加图例
# # plt.legend()
#
# # 显示图形
# plt.show()
