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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


med_tls_em_rmse0=[0.06598929332466051, 0.06599355571012884, 0.06600940746484218, 0.06602314425273148, 0.06607245667730186, 0.06609270515492059, 0.06612312620389788, 0.06603702947117687, 0.06614513653009517, 0.06612890020956724, 0.06620268737823926, 0.06639075374706453, 0.06644839196479878]
med_tls_rmse0=[0.06640386578688108, 0.06670189019096182, 0.06708430795577291, 0.06788450383996647, 0.06907296513397307, 0.07047379260760848, 0.0718296914982664, 0.07327211168801523, 0.07526516662976285, 0.07807148794882005, 0.08008003052394821, 0.08262438741103093, 0.08659801692422611]
med_ls_em_rmse0=[0.06673169245405872, 0.06673386481155735, 0.06675555542086531, 0.06670424584528985, 0.06675229288905525, 0.06678157991345748, 0.06679339111619526, 0.06680473928532506, 0.0667072160932767, 0.06671867000162525, 0.06676462700934402, 0.06679477287237409, 0.06668739392807033]
med_ls_rmse0=[0.06998954594914816, 0.07242562460623732, 0.07532032989839546, 0.07886993863786168, 0.08249534451685334, 0.0861185618369131, 0.09005102238464094, 0.09403095773738054, 0.09819822974299963, 0.10177120467397832, 0.10574711326067462, 0.10918993582623569, 0.11291325313642755]


med_tls_em_rmse1=[0.06604722710775728, 0.06600834884645429, 0.06597303424371151, 0.06603954439848159, 0.06604002290191827, 0.066119962230102, 0.06606330779753075, 0.06614071897957316, 0.06619455551435605, 0.06624368409136236, 0.06627585600456531, 0.0662634693920873, 0.06639072167230496]
med_tls_rmse1=[0.06640422643342431, 0.06655677815861313, 0.06703681001868199, 0.0676302266363388, 0.06876351975612677, 0.06986480244441773, 0.07122362908090552, 0.07286896401620248, 0.07456482217447918, 0.0766778550118089, 0.07895513336026855, 0.08115970741019943, 0.08382528787288154]
med_ls_em_rmse1=[0.06682926762820265, 0.06682695844263073, 0.06687541861988855, 0.06688247410011047, 0.0668943405988175, 0.06694406051260424, 0.06693666175729926, 0.06685481691434189, 0.06686159291728361, 0.06694444051504399, 0.06699535744088198, 0.06699971636829877, 0.06700499959855351]
med_ls_rmse1=[0.06951859733069743, 0.071803285291872, 0.07441900097273887, 0.07780791018618731, 0.08100857504335748, 0.08452290435003332, 0.08835752444426852, 0.0921801249359055, 0.09633054022823434, 0.09991538666290192, 0.10342752087203179, 0.10737690122413895, 0.11081071069602888]


med_tls_em_rmse2=[0.06603171725804166, 0.06597807737953162, 0.06594151693582094, 0.06593007735010331, 0.06594040346345587, 0.06593114645950934, 0.06590390580818992, 0.06589774381502117, 0.06587628033419643, 0.06587321357841784, 0.06586847389337958, 0.06587151799648074, 0.06588007249432087]
med_tls_rmse2=[0.06636797573078668, 0.06660309574641401, 0.06716639684160072, 0.06784644750945325, 0.0691031466552818, 0.07037547837628344, 0.07177711352665392, 0.07336093356140373, 0.07522422617801086, 0.0774599354996948, 0.07956147830094909, 0.08185491686166949, 0.08517428106867779]
med_ls_em_rmse2=[0.06686043369125308, 0.06687441950273115, 0.06696864568727617, 0.06695757938405161, 0.06695945690983598, 0.06698265484752697, 0.06700309182678835, 0.06699635254151609, 0.06698891299622908, 0.06697891544491458, 0.06699021837843028, 0.06699199515785297, 0.06703144022454033]
med_ls_rmse2=[0.06967664477407448, 0.07206526526284504, 0.07491311565349582, 0.07844334602360215, 0.08182942845665392, 0.0854329664320327, 0.08923297111864961, 0.09321656914488141, 0.09734240660712996, 0.10107091141425398, 0.10493634232353859, 0.10857473730194556, 0.11199069232288651]


med_tls_em_rmse3=[0.06623570118741232, 0.06622181220963469, 0.06621715253808491, 0.0662906347243804, 0.0663793041398368, 0.06636801703053553, 0.06640774343019866, 0.06642532382610672, 0.06656546691400553, 0.06666568009036483, 0.06671219503339348, 0.06675657459889789, 0.06680828777957906]
med_tls_rmse3=[0.06636377952187601, 0.0665687013597557, 0.06714963897312054, 0.06805465683645565, 0.06909250478329304, 0.07062141278035525, 0.071899873060402, 0.07354346587004124, 0.07554930939152729, 0.0777624863848847, 0.08018291873047781, 0.0826075988174104, 0.08627768257409732]
med_ls_em_rmse3=[0.0668762742466962, 0.06689342995265679, 0.06694073405338305, 0.06694046551682477, 0.06696234274138702, 0.0669522246317486, 0.06702516437165021, 0.06703611971020826, 0.06706738486279525, 0.06713735170536686, 0.06713313377696917, 0.06720737250067137, 0.06727793189633743]
med_ls_rmse3=[0.06995510086440221, 0.07238848432693729, 0.07550628177750114, 0.07894923762227941, 0.08239575735526022, 0.0862732984867343, 0.09003761091177023, 0.09413091778612673, 0.09811385810409534, 0.10198976874998011, 0.10602850703010715, 0.10947327798835975, 0.11286986033868754]

x_plt = train_array = np.arange(0.4, 1.02, 0.05)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 在每个子图中绘制对应的数据
axs[0, 0].plot(x_plt, med_tls_em_rmse0,label='TLS_EM',marker='^', linestyle='-' )
axs[0, 0].plot(x_plt, med_tls_rmse0,label='TLS',marker='D', linestyle='-')
axs[0, 0].plot(x_plt, med_ls_em_rmse0,label='LS_EM',marker='*', linestyle='-' )
axs[0, 0].plot(x_plt, med_ls_rmse0,label='LS',marker='o', linestyle='-' )
# axs[0, 0].set_xlabel('Noise')
# axs[0, 0].set_ylabel('RMSE')
# axs[0, 0].set_xticks(x_plt)
axs[0, 0].set_title('[1, 0.1, 0.02]')
axs[0, 0].locator_params(axis='x', nbins=10)  # 在横坐标上显示6个刻度
axins = inset_axes(axs[0, 0], width="20%", height="20%", loc='lower right',bbox_to_anchor=(0.01, 0.07, 1, 1),
                   bbox_transform=axs[0,0].transAxes)
# 在放大区域中绘制数据
axins.plot(np.arange(0,6,1), med_tls_em_rmse0[6:12],label='TLS_EM',linestyle='-',marker='^',color='blue' )
axins.plot(np.arange(0,6,1), med_ls_em_rmse0[6:12],label='LS_EM',linestyle='-',marker='*',color='green' )
# 不显示放大区域的坐标刻度
axins.set_xticks([])
axins.set_yticks([])


# axs[0, 0].title("data split:{},noise generation:{}".format(s, m))

axs[0, 1].plot(x_plt, med_tls_em_rmse1,label='TLS_EM',marker='^', linestyle='-')
axs[0, 1].plot(x_plt, med_tls_rmse1, label='TLS',marker='D', linestyle='-')
axs[0, 1].plot(x_plt, med_ls_em_rmse1, label='LS_EM',marker='*', linestyle='-')
axs[0, 1].plot(x_plt, med_ls_rmse1, label='LS',marker='o', linestyle='-')
# axs[0, 1].set_xlabel('Noise')
# axs[0, 1].set_ylabel('RMSE')
axs[0, 1].set_title('[0.97, 0.07, 0.05]')
axs[0, 1].set_xticks(x_plt)
axs[0, 1].locator_params(axis='x', nbins=10)  # 在横坐标上显示6个刻度

axins = inset_axes(axs[0, 1], width="20%", height="20%", loc='lower right',bbox_to_anchor=(0.01, 0.07, 1, 1),
                   bbox_transform=axs[0,1].transAxes)
# 在放大区域中绘制数据
axins.plot(np.arange(0,6,1), med_tls_em_rmse1[6:12],label='TLS_EM',linestyle='-',marker='^',color='blue' )
axins.plot(np.arange(0,6,1), med_ls_em_rmse1[6:12],label='LS_EM',linestyle='-',marker='*',color='green' )
# 不显示放大区域的坐标刻度
axins.set_xticks([])
axins.set_yticks([])



axs[1, 0].plot(x_plt, med_tls_em_rmse2, label='TLS_EM',marker='^', linestyle='-' )
axs[1, 0].plot(x_plt, med_tls_rmse2, label='TLS',marker='D', linestyle='-')
axs[1, 0].plot(x_plt, med_ls_em_rmse2, label='LS_EM',marker='*', linestyle='-' )
axs[1, 0].plot(x_plt, med_ls_rmse2, label='LS',marker='o', linestyle='-')
axs[1, 0].set_title('[0.99, 0.04, 0.03]')
axs[1, 0].set_xticks(x_plt)
axs[1, 0].locator_params(axis='x', nbins=10)  # 在横坐标上显示6个刻度
axins = inset_axes(axs[1, 0], width="20%", height="20%", loc='lower right',bbox_to_anchor=(0.01, 0.07, 1, 1),
                   bbox_transform=axs[1,0].transAxes)
# 在放大区域中绘制数据
axins.plot(np.arange(0,6,1), med_tls_em_rmse2[6:12],label='TLS_EM',linestyle='-',marker='^',color='blue' )
axins.plot(np.arange(0,6,1), med_ls_em_rmse2[6:12],label='LS_EM',linestyle='-',marker='*',color='green' )
# 不显示放大区域的坐标刻度
axins.set_xticks([])
axins.set_yticks([])


axs[1, 1].plot(x_plt, med_tls_em_rmse3, label='TLS_EM',marker='^', linestyle='-' )
axs[1, 1].plot(x_plt, med_tls_rmse3, label='TLS',marker='D', linestyle='-')
axs[1, 1].plot(x_plt, med_ls_em_rmse3, label='LS_EM' ,marker='*', linestyle='-')
axs[1, 1].plot(x_plt, med_ls_rmse3, label='LS',marker='o', linestyle='-')
# axs[1, 1].set_xlabel('Noise')
# axs[1, 1].set_ylabel('RMSE')
axs[1, 1].set_title('[1,0.06,0.08]')
axs[1, 1].set_xticks(x_plt)
axs[1, 1].locator_params(axis='x', nbins=10)  # 在横坐标上显示6个刻度
axins = inset_axes(axs[1, 1], width="20%", height="20%", loc='lower right',bbox_to_anchor=(0.01, 0.07, 1, 1),
                   bbox_transform=axs[1,1].transAxes)
# 在放大区域中绘制数据
axins.plot(np.arange(0,6,1), med_tls_em_rmse3[6:12],label='TLS_EM',linestyle='-',marker='^',color='blue' )
axins.plot(np.arange(0,6,1), med_ls_em_rmse3[6:12],label='LS_EM',linestyle='-',marker='*',color='green' )
# 不显示放大区域的坐标刻度
axins.set_xticks([])
axins.set_yticks([])

# 添加标题和标签
plt.subplots_adjust(wspace=0.5, hspace=0.5)
# 在整个图上添加 'a', 'b', 'c', 'd' 的标注
fig.text(0.05, 0.9, 'a', fontsize=14, fontweight='bold')
fig.text(0.5, 0.9, 'b', fontsize=14, fontweight='bold')
fig.text(0.05, 0.46, 'c', fontsize=14, fontweight='bold')
fig.text(0.5, 0.46, 'd', fontsize=14, fontweight='bold')

# fig.suptitle('Comparison of RMSE for Four Methods')
for ax in axs.flat:
    ax.set(xlabel='train_size', ylabel='RMSE')
    ax.legend()

plt.show()
