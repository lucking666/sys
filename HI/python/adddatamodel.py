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
from saved_xgb_regression_model import OptimizedXGBRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
import copy
import scipy.stats as stats

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


X_train, X_test = feature2_scale[:99, :], feature2_scale[99:, :]
y_train, y_test = labellog[:99], labellog[99:]






def add_noise(arr, std_dev):
    random.seed(std_dev)
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
def interaction_subtract(x):
    n_features = x.shape[1]

    # 初始化一个空的列表来存储相减列
    subtracted_cols = []

    # 使用两个嵌套循环来计算每一对特征之间的差值
    for i in range(n_features):
        for j in range(i, n_features):
            subtracted_col = x[:, i] - x[:, j]
            subtracted_cols.append(subtracted_col)

    # 将原始列和相减列合并成新数组 X_
    X_ = np.column_stack([x] + subtracted_cols)

    return X_
def ceemdan(X_train, X_test):
    X_data = np.vstack((X_train, X_test))
    IImfs = []
    data = copy.deepcopy(X_data).ravel()
    ceemdan= CEEMDAN()
    ceemdan.trials = 100  # 迭代次数
    ceemdan.max_siftings = 50  # SIFT 迭代次数
    ceemdan.noise_std = 0.01  # 白噪声标准差
    ceemdan.ensemble_size=5

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
    new_data=result_array=np.hstack((X_data, np.transpose(IImfs)))
    X_train, X_test = new_data[:99, :], new_data[99:, :]
    X_train = interaction_subtract(X_train)
    X_test = interaction_subtract(X_test)
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

    # 创建OptimizedXGBRegressor对象
    xgb_model = OptimizedXGBRegressor()

    # 拟合数据
    xgb_model.fit(X_train, y_train)

    # 预测
    y_pred = xgb_model.predict(X_test)

    mae, rmse = evaluation(y_test, y_pred)
    # print('mae——{},rmse——{}'.format(mae, rmse))
    return mae, rmse


def interaction(x):
    n_features = x.shape[1]
    combinations = [(i, j) for i in range(n_features) for j in range(i, n_features)]
    interaction_cols = [x[:, i] * x[:, j] for i, j in combinations]

    # 将原始列和相互作用列合并成新数组 X_
    X_ = np.column_stack([x] + interaction_cols)

    return X_



def showFeatures(data):
    x = range(58)

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
    selected_feature_indices = np.array(np.where(np.abs(feature_importances)>0.4)).ravel()
    # 打印被选择的特征的索引和特征值
    print("被选择的特征的索引：", selected_feature_indices)
    selected_features = X[:, selected_feature_indices]
    # print("被选择的特征：", selected_features)
    X=selected_features
    selected_columns = Xtest[:, selected_feature_indices]



    return X,selected_columns,selected_feature_indices



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

    print("特征权重:", weights)
    w=weights
    # w=np.abs(weights)

    X_train=X*w
    X__test=X_test*w


    return X_train,X__test


def getRFE_RFfeatures(X,Y,X_test):
    model = RandomForestRegressor()

    # 创建特征递归消除对象
    rfe = RFECV(model,min_features_to_select=1,)  # 选择3个最重要的特征

    # 使用特征递归消除选择特征
    rfe.fit(X, Y.ravel())

    # 获取所选择的特征列索引值
    selected_feature_indices = np.where(rfe.ranking_ == 1)[0]

    print("选择的特征列索引值:", selected_feature_indices)

    X_train=X[:,selected_feature_indices]
    X__test=X_test[:,selected_feature_indices]


    return X_train,X__test

def calculate(X,Y):
    r,p=stats.spearmanr(X,Y)
    return r


maelist = []
rmselist = []
maelistnoise = []
rmselistnoise = []
maelistnoiseemd = []
rmselistnoiseemd = []
maelistnoiseemd1 = []
rmselistnoiseemd1 = []
for i in range(10):
    random.seed(i)
    Xtrain = X_train
    Xtest = X_test
    mae, rmse = get_result(Xtrain, y_train, Xtest, y_test)
    std = random.uniform(0.2, 0.8)

    # std = 0.1

    # Xtrain = add_noise(Xtrain, std)#加噪声
    maenoise, rmsenoise = get_result(Xtrain, y_train, Xtest, y_test)
    maelistnoise.append(maenoise)
    rmselistnoise.append(rmsenoise)

    Xtrain, Xtest = ceemdan(Xtrain, Xtest)

    celllist = []
    for col in range(len(Xtrain[0])):
        column_data = Xtrain[:, col]
        celllist.append(calculate(column_data, y_train))
    print(celllist)
    Xtrain__, Xtest__, index = getRFfeatures(Xtrain, y_train, Xtest)  # 随机森林
    Xtrain = np.delete(Xtrain, index, axis=1)
    Xtest = np.delete(Xtest, index, axis=1)
    # Xtrain, Xtest = getReliefFfeatures(Xtrain, y_train, Xtest)#reliefF算法
    Xtrain, Xtest = getRFE_RFfeatures(Xtrain, y_train, Xtest)#特征递归消除和随机森林结合
    Xtrain=np.hstack((Xtrain__,Xtrain))
    Xtest=np.hstack((Xtest__,Xtest))
    # print("到这里是模态分解完毕,使用随机森林进行特征选择,得到的结果作为最终结果")
    maenoiseemd, rmsenoiseemd = get_result(Xtrain, y_train, Xtest, y_test)
    maelist.append(mae)
    rmselist.append(rmse)
    maelistnoiseemd.append(maenoiseemd)
    rmselistnoiseemd.append(rmsenoiseemd)
    print(std,maenoiseemd,rmsenoiseemd)

print('原始mae——{},rmse——{}'.format(np.median(maelist), np.median(rmselist)))
print('加噪声mae——{},rmse——{}'.format(np.median(maelistnoise), np.median(rmselistnoise)))
print('加噪声em加特征选择算法mae——{},rmse——{}'.format(np.median(maelistnoiseemd), np.median(rmselistnoiseemd)))
print(maelistnoiseemd,rmselistnoiseemd)

# 所有数据
# RFE_RF(1):
# 原始mae——0.09008838082636496,rmse——0.12103350911903503
# 加噪声mae——0.20373386011910097,rmse——0.2517368501350008
# 加噪声em加特征选择算法mae——0.16532381880320945,rmse——0.2020038337791155RFE_RF
# 加噪声em加特征选择算法mae——0.11520552147132831,rmse——0.15554476839948772RF
# 加噪声em加特征选择算法mae——0.11306274689183873,rmse——0.14884054003576788reliefF

# 原始mae——0.12837467958092807,rmse——0.16247461710768774
# 加噪声mae——0.19537310080244222,rmse——0.2525831752896243
# 加噪声em加特征选择算法mae——0.11410956288137784,rmse——0.1518677213213217随机森林前三
# 加噪声em加特征选择算法mae——0.11792076763045939,rmse——0.15646767284988014reliefF算法


# 原始mae——0.12837467958092807,rmse——0.16247461710768774
# 加噪声mae——0.25750862829506616,rmse——0.28988990684706284
# 加噪声em加特征选择算法mae——0.10816940744928091,rmse——0.14180057934280685RF_RFECV
