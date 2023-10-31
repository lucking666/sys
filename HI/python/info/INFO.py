import numpy as np

def INFO(nP, MaxIt, lb, ub, dim, fobj):
    def BC(X, lb, ub):
        Flag4ub = X > ub
        Flag4lb = X < lb
        X = (X * (~(Flag4ub + Flag4lb))) + ub * Flag4ub + lb * Flag4lb
        return X

    # Initialization
    Cost = np.zeros(nP)
    M = np.zeros(nP)

    X = initialization(nP, dim, ub, lb)
    
    for i in range(nP):
        Cost[i] = fobj(X[i, :])
        M[i] = Cost[i]

    ind = np.argsort(Cost)
    Best_X = X[ind[0], :]
    Best_Cost = Cost[ind[0]]

    Worst_Cost = Cost[ind[-1]]
    Worst_X = X[ind[-1], :]

    I = np.random.randint(2, 5)
    Better_X = X[ind[I], :]
    Better_Cost = Cost[ind[I]]

    Convergence_curve = []

    # Main Loop of INFO
    for it in range(MaxIt):
        alpha = 2 * np.exp(-4 * (it / MaxIt))
        M_Best = Best_Cost
        M_Better = Better_Cost
        M_Worst = Worst_Cost

        for i in range(nP):
            del_val = 2 * np.random.rand() * alpha - alpha
            sigm = 2 * np.random.rand() * alpha
            A1 = np.random.permutation(nP)
            A1 = A1[A1 != i]
            a, b, c = A1[0], A1[1], A1[2]

            e = 1e-25
            epsi = e * np.random.rand()

            omg = max([M[a], M[b], M[c]])
            MM = [M[a] - M[b], M[a] - M[c], M[b] - M[c]]

            W = [np.cos(MM[0] + np.pi) * np.exp(-abs(MM[0] / omg)),
                 np.cos(MM[1] + np.pi) * np.exp(-abs(MM[1] / omg)),
                 np.cos(MM[2] + np.pi) * np.exp(-abs(MM[2] / omg))]

            Wt = sum(W)
            WM1 = del_val * (W[0] * (X[a, :] - X[b, :]) + W[1] * (X[a, :] - X[c, :]) + W[2] * (X[b, :] - X[c, :])) / (
                    Wt + 1) + epsi

            omg = max([M_Best, M_Better, M_Worst])
            MM = [M_Best - M_Better, M_Best - M_Better, M_Better - M_Worst]

            W = [np.cos(MM[0] + np.pi) * np.exp(-abs(MM[0] / omg)),
                 np.cos(MM[1] + np.pi) * np.exp(-abs(MM[1] / omg)),
                 np.cos(MM[2] + np.pi) * np.exp(-abs(MM[2] / omg))]

            Wt = sum(W)
            WM2 = del_val * (W[0] * (Best_X - Better_X) + W[1] * (Best_X - Worst_X) + W[2] * (
                    Better_X - Worst_X)) / (Wt + 1) + epsi

            r = np.random.uniform(0.1, 0.5)
            MeanRule = r * WM1 + (1 - r) * WM2

            if np.random.rand() < 0.5:
                z1 = X[i, :] + sigm * (np.random.rand() * MeanRule) + np.random.randn() * (
                        Best_X - X[a, :]) / (M_Best - M[a] + 1)
                z2 = Best_X + sigm * (np.random.rand() * MeanRule) + np.random.randn() * (X[a, :] - X[b, :]) / (
                        M[a] - M[b] + 1)
            else:
                z1 = X[a, :] + sigm * (np.random.rand() * MeanRule) + np.random.randn() * (X[b, :] - X[c, :]) / (
                        M[b] - M[c] + 1)
                z2 = Better_X + sigm * (np.random.rand() * MeanRule) + np.random.randn() * (X[a, :] - X[b, :]) / (
                        M[a] - M[b] + 1)

            u = np.zeros(dim)
            for j in range(dim):
                mu = 0.05 * np.random.randn()
                if np.random.rand() < 0.5:
                    if np.random.rand() < 0.5:
                        u[j] = z1[j] + mu * abs(z1[j] - z2[j])
                    else:
                        u[j] = z2[j] + mu * abs(z1[j] - z2[j])
                else:
                    u[j] = X[i, j]

            if np.random.rand() < 0.5:
                L = np.random.rand() < 0.5
                v1 = (1 - L) * 2 * np.random.random() + L
                v2 = np.random.random() * L + (1 - L)
                Xavg = (X[a, :] + X[b, :] + X[c, :]) / 3
                phi = np.random.random()
                Xrnd = phi * (Xavg) + (1 - phi) * (phi * Better_X + (1 - phi) * Best_X)
                Randn = L * np.random.randn(dim) + (1 - L) * np.random.randn()
                if np.random.rand() < 0.5:
                    u = Best_X + Randn * (MeanRule + np.random.randn() * (Best_X - X[a, :]))
                else:
                    u = Xrnd + Randn * (MeanRule + np.random.randn() * (v1 * Best_X - v2 * Xrnd))

            New_X = BC(u, lb, ub)
            New_Cost = fobj(New_X)

            if New_Cost < Cost[i]:
                X[i, :] = New_X
                Cost[i] = New_Cost
                M[i] = Cost[i]
                if Cost[i] < Best_Cost:
                    Best_X = X[i, :]
                    Best_Cost = Cost[i]

        ind = np.argsort(Cost)
        Worst_X = X[ind[-1], :]
        Worst_Cost = Cost[ind[-1]]
        I = np.random.randint(2, 5)
        Better_X = X[ind[I], :]
        Better_Cost = Cost[ind[I]]

        Convergence_curve.append(Best_Cost)

        print(f'Iteration {it + 1}, Best Cost = {Best_Cost}')

    return Best_Cost, Best_X, Convergence_curve

# 以下是 initialization 函数的 Python 版本
def initialization(nP, dim, ub, lb):
    Boundary_no = len(ub)
    X = np.zeros((nP, dim))

    if Boundary_no == 1:
        X = np.random.rand(nP, dim) * (ub - lb) + lb

    if Boundary_no > 1:
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            X[:, i] = np.random.rand(nP) * (ub_i - lb_i) + lb_i

    return X
