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





items = ['F2', 'F5', 'F3']


# tls和em算法结合


# 加载数据
data_all = pd.read_csv('dataset.csv')
data = data_all[['F2', 'F5', 'F3', 'cyclelife']]  # 注意特征与feature_remain保持一致!!!
_class = [0] * 41 + [1] * 43 + [2] * 40
data['class'] = _class
_xita = [0] * 124
data['xita'] = _xita

x_plt = np.arange(0, 41, 1)
y1=data['F2'].iloc[:41]
y2=data['F2'].iloc[:41]
# 生成均值为0，标准差为5的高斯噪声
noise = np.random.normal(loc=0, scale=0.04, size=len(y2))

# 将噪声添加到y2中
y2_with_noise = y2 + noise
plt.scatter(x_plt, y1, label='true value')
plt.scatter(x_plt, y2_with_noise, label='measured value')
# plt.plot(x_plt, med_tls_rmse)
# plt.plot(x_plt, med_ls_em_rmse, )
# plt.plot(x_plt, med_ls_rmse)
# plt.legend(['TLS_EM', 'TLS', 'LS_EM', 'LS'])  #
plt.xlabel('cell')
plt.ylabel('voltage')
plt.xticks(x_plt)
plt.legend()
plt.locator_params(axis='x', nbins=10)
plt.show()



