
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

med_tls_em_rmse=[0.25128163511572194, 0.1546771512187762, 0.11847141309077967, 0.10857369105364642, 0.10696324746086816, 0.10722517913737863, 0.1109779051228727, 0.08934234500562284, 0.0806857143908698]
med_tls_rmse=[0.3531222765888313, 0.29657082202291885, 0.24893167566108212, 0.19075662256325315, 0.16474051837412795, 0.14994977784056113, 0.13074508430646126, 0.10151525663052549, 0.08722431970691497]

train_array = np.arange(0.15, 1.0, 0.1)
x_plt=train_array
plt.plot(x_plt,med_tls_em_rmse,marker='o', linestyle='-')
plt.plot(x_plt,med_tls_rmse,marker='o', linestyle='-')
plt.legend(['TLS_EM', 'TLS']) #
plt.xticks(x_plt)
plt.xlabel('train_size')
plt.ylabel('RMSE')
plt.show()