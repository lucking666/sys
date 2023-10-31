import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 定义目标函数
def fobj(x):
    # 在这里定义你的目标函数（Func_name='F10'）
    # 请确保将函数返回值设置为目标函数的值

    return ...

# 定义优化参数
nP = 30  # Number of Population
lb, ub = -1, 1  # 设置搜索空间的下限和上限
dim = 10  # 维度

# 初始化最佳位置和适应度
BestPositions = np.zeros(dim)
Best_fitness = np.inf
Convergence_curve = []

# 优化函数
def on_iteration(x):
    global BestPositions, Best_fitness, Convergence_curve
    fitness = fobj(x)
    if fitness < Best_fitness:
        Best_fitness = fitness
        BestPositions = x
    Convergence_curve.append(Best_fitness)

# 运行优化
result = minimize(fobj, np.random.uniform(lb, ub, dim), method='Nelder-Mead', callback=on_iteration)

# 绘制收敛曲线
plt.figure()
plt.semilogy(Convergence_curve, color='r', linewidth=4)
plt.title('Convergence curve')
plt.xlabel('Iteration')
plt.ylabel('Best fitness obtained so far')
plt.axis('tight')
plt.grid(False)
plt.box(True)
plt.legend(['INFO'])
plt.show()