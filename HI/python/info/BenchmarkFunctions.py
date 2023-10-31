import numpy as np

# F1
def F1(x):
    return np.sum(x**2)

# F2
def F2(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

# F3
def F3(x):
    dim = x.shape[0]
    total = 0
    for i in range(dim):
        total += np.sum(x[:i+1])**2
    return total

# F4
def F4(x):
    return np.max(np.abs(x))

# F5
def F5(x):
    dim = x.shape[0]
    total = 0
    for i in range(1, dim):
        total += 100 * (x[i] - x[i-1]**2)**2 + (x[i-1] - 1)**2
    return total

# F6
def F6(x):
    return np.sum(np.abs(x + 0.5)**2)

# F7
def F7(x):
    dim = x.shape[0]
    total = 0
    for i in range(dim):
        total += (i + 1) * x[i]**4
    total += np.random.rand()
    return total

# F8
def F8(x):
    return -np.sum(x * np.sin(np.sqrt(np.abs(x))))

# F9
def F9(x):
    dim = x.shape[0]
    total = np.sum(x**2 - 10 * np.cos(2 * np.pi * x)) + 10 * dim
    return total

# F10
def F10(x):
    dim = x.shape[0]
    total = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / dim)) - \
        np.exp(np.sum(np.cos(2 * np.pi * x)) / dim) + 20 + np.exp(1)
    return total

# F11
def F11(x):
    dim = x.shape[0]
    total = np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, dim + 1))) + 1)
    return total

# F12
def F12(x):
    dim = x.shape[0]
    a = np.array([3, 10, 30, 10])
    c = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747],
                  [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]])
    total = np.sum((a * (x - pH)**2), axis=1) + c
    return -0.1 * np.prod(np.exp(-total))

# F13
def F13(x):
    dim = x.shape[0]
    a = np.array([4, 10, 30, 10])
    c = np.array([0.1, 0.2, 0.2, 0.4])
    pH = np.array([[0.1312, 0.1696, 0.5569], [0.2329, 0.4135, 0.8307],
                  [0.2348, 0.1415, 0.3522], [0.4047, 0.8828, 0.8732]])
    total = np.sum((a * (x - pH)**2), axis=1) + c
    return -0.1 * np.prod(np.exp(-total))

# F14
def F14(x):
    aS = np.array([[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
                   [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]])
    total = np.sum((x - aS)**6, axis=0)
    return (1 / 500 + np.sum(1 / (1 + total)))**(-1)

# F15
def F15(x):
    aK = np.array([0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
    bK = 1 / np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
    total = np.sum((aK - (x[0] * (bK**2 + x[1] * bK) / (bK**2 + x[2] * bK + x[3]))**2))
    return total
# F16
def F16(x):
    total = 4 * x[0]**2 - 2.1 * x[0]**4 + x[0]**6 / 3 + x[0] * x[1] - 4 * x[1]**2 + 4 * x[1]**4
    return total

# F17
def F17(x):
    total = (x[1] - (x[0]**2) * 5.1 / (4 * (3.14159265359**2)) + 5 / 3.14159265359 * x[0] - 6)**2 + 10 * (1 - 1 / (8 * 3.14159265359)) * np.cos(x[0]) + 10
    return total

# F18
def F18(x):
    total = (1 + (x[0] + x[1] + 1)**2 * (19 - 14 * x[0] + 3 * x[0]**2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1]**2)) * (30 + (2 * x[0] - 3 * x[1])**2 * (18 - 32 * x[0] + 12 * x[0]**2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1]**2))
    return total

# F19
def F19(x):
    aH = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747], [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]])
    total = 0
    for i in range(4):
        total -= cH[i] * np.exp(-np.sum(aH[i, :] * ((x - pH[i, :])**2)))
    return total

# F20
def F20(x):
    aH = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
    cH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    pH = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886], [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                   [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650], [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
    total = 0
    for i in range(4):
        total -= cH[i] * np.exp(-np.sum(aH[i, :] * ((x - pH[i, :])**2)))
    return total

# F21
def F21(x):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7],
                    [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    total = 0
    for i in range(5):
        total -= ((x - aSH[i, :]).dot(x - aSH[i, :]) + cSH[i])**(-1)
    return total

# F22
def F22(x):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7],
                    [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    total = 0
    for i in range(7):
        total -= ((x - aSH[i, :]).dot(x - aSH[i, :]) + cSH[i])**(-1)
    return total

# F23
def F23(x):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7],
                    [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    total = 0
    for i in range(10):
        total -= ((x - aSH[i, :]).dot(x - aSH[i, :]) + cSH[i])**(-1)
    return total


