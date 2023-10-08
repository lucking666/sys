for j in range(n):  # 调整噪声大小
    np.random.seed(j)
    print("noise_level:", j)
    tls_rmse = []
    ls_rmse = []
    tls_em_rmse = []
    ls_em_rmse = []

    times[0] = (times[0] * j * 0.05)
    times[1] = (times[1] * j * 0.05)
    times[2] = (times[2] * j * 0.05)
    copy_data = data_x

    for p in range(s):
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

        data_train_random[..., 5] = np.log10(data_train_random[..., 5])  # dataframe
        X_test_random[..., 5] = np.log10(X_test_random[..., 5])  # dataframe
        X_test_random = np.random.permutation(X_test_random)
        Y_test_random = X_test_random[..., 5].reshape(-1, 1)

        for k in range(m):  # 生成m次噪声
            # print("不同噪声矩阵：")

            X_train = copy.deepcopy(data_train_random)
            # Y_train = copy.deepcopy(data_train_random.iloc)
            X_test_random = X_test_random[..., 0:5]
            X_test = copy.deepcopy(X_test_random)  # dataframe
            Y_test = copy.deepcopy(Y_test_random)
            standard_X = np.std(X_train, axis=0)  # .reshape(-1, 1)
            np.random.seed(k)
            noise_X0 = np.random.normal(loc=0, scale=times[0], size=(N_train[0], 6))
            noise_X1 = np.random.normal(loc=0, scale=times[1], size=(N_train[1], 6))
            noise_X2 = np.random.normal(loc=0, scale=times[2], size=(N_train[2], 6))

            noise_X = np.concatenate((np.concatenate((noise_X0, noise_X1), axis=0), noise_X2), axis=0)
            # print('noise_X is :',noise_X[0])
            # print('noise_X is :',noise_X[50])

            X_train_noise = copy.deepcopy(X_train)
            Y_train = X_train_noise[..., 5].reshape(-1, 1)

            for index in range(len(X_train_noise)):
                flag = int(X_train_noise[index][6])
                for i in range(6):
                    noise_X[:, i] *= (standard_X[i])  # 根据每个特征的标准差生成噪声
                    X_train_noise[:, i] += noise_X[:, i]

            # 转换数据类型（前面使用DataFrame是因为之前要进行特征选择）
            x_train = X_train_noise
            Y_train_noise = np.array(X_train_noise[:, 5]).reshape(-1, 1)
            x_test = X_test

            # # 总体最小二乘
            # W_tls, b_tls, = tls(x_train[:,0:5], Y_train_noise)#x:array,y:array,(-1,1)
            # W_tls_em,b_tls_em=add_em(x_train, Y_train_noise,'tls',x_test,Y_test)#x:array,y:array,(-1,1)
            # y_pred_tls = np.dot(x_test, W_tls) + b_tls
            # y_pred_tls_em = np.dot(x_test, W_tls_em) + b_tls_em
            # tls_rmse.append(rmse(Y_test, y_pred_tls))
            # tls_em_rmse.append(rmse(Y_test, y_pred_tls_em))
            # print('tls_em rmse and tls rmse is:',rmse(Y_test, y_pred_tls_em),rmse(Y_test, y_pred_tls))

            # print("x_train",x_train[0])
            # 最小二乘
            W_ls, b_ls, = ls(x_train[:, 0:5], Y_train_noise)
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