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
#
#
# import numpy as np
# import matplotlib.gridspec as gridspec
# data_all = pd.read_csv('dataset.csv')
# data = data_all[['F2', 'F3', 'F5','cyclelife']]  # 注意特征与feature_remain保持一致!!!
# _class = [0] * 41 + [1] * 43 + [2] * 40
# data['class']=_class
#
#
# # 获取不同 class 值的颜色映射
# colors = {0: 'red', 1: 'green', 2: 'blue'}
#
# # 创建 1 行 3 列的图，确保每个子图是正方形
# fig = plt.figure(figsize=(15, 5))
# gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
#
# # 遍历 'F2', 'F3', 'F5'
# for i, feature in enumerate(['F2', 'F3', 'F5']):
#     # 在每个子图中绘制散点图
#     axs = plt.subplot(gs[i])
#     for cls, color in colors.items():
#         subset = data[data['class'] == cls]
#         axs.scatter(subset.index, subset[feature], label=f'class {cls}', color=color)
#
#     # 设置图标题和标签
#     axs.set_xlabel('battery cell', fontsize=12)
#     axs.set_ylabel(feature, fontsize=12)
#     axs.legend()
# fig.text(0.01, 0.93, 'a', fontsize=14, fontweight='bold')
# fig.text(0.33, 0.93, 'b', fontsize=14, fontweight='bold')
# fig.text(0.68, 0.93, 'c', fontsize=14, fontweight='bold')
# # 调整布局
# plt.subplots_adjust(top=0.9, hspace=0.6, wspace=0.4, left=0.05, right=0.95)  # 调整顶部留白、子图之间的垂直间距、水平间距，左右留白
# # plt.subplots_adjust(top=0.9, hspace=0.4, left=0.05, right=0.95)  # 调整顶部留白、子图之间的垂直间距，左右留白
# plt.show()
#
# # med_tls_em_rmse0=[0.10739202428228475, 0.09372489429832595, 0.08894437664393298, 0.08746814354927684, 0.0805287675551736, 0.07631105441331074, 0.07256498038830031, 0.07074661682297248, 0.0663424153361223]
# # med_tls_rmse0=[0.24180531239573386, 0.16726337760792784, 0.14268178893446404, 0.12819288298057943, 0.11059570043537237, 0.1040410916615759, 0.09650127572830089, 0.08913648878641837, 0.08037076053834957]
# # med_ls_em_rmse0=[0.10334282496788533, 0.0933980035248172, 0.08953837814079055, 0.08873002092683008, 0.08316023899237245, 0.07869102788456843, 0.07476660612033915, 0.07262161465731717, 0.06874818650035505]
# # med_ls_rmse0=[0.14613540381067502, 0.12957470364233442, 0.12318252882292714, 0.12122241825884233, 0.11730007306777657, 0.11666292465177139, 0.11516532573677707, 0.11273416264047069, 0.10530821390728343]
# #
# # med_tls_em_rmse1=[0.10750407384036514, 0.09313456151863742, 0.08847369750602299, 0.08684443193081057, 0.08010939249141089, 0.07594077763422255, 0.07224368780480583, 0.07050003744125236, 0.06590492866487202]
# # med_tls_rmse1=[0.23245050554072022, 0.16105460024607712, 0.1375666729559965, 0.12391934545244487, 0.10746350008962349, 0.10136755951274085, 0.09417552861227357, 0.08724217841921192, 0.07880901756525664]
# # med_ls_em_rmse1=[0.10361283551172809, 0.09289698543982464, 0.0892996430008718, 0.08852262389668891, 0.08322636921274051, 0.0787096265745818, 0.07458131456294631, 0.0723418787881559, 0.06836187674308145]
# # med_ls_rmse1=[0.14338418659904167, 0.12733680019653149, 0.12095816882492791, 0.11908499836119471, 0.11528325844198253, 0.11469120464308288, 0.1132542140899995, 0.11081098686482148, 0.10347172825018021]
# #
# #
# # med_tls_em_rmse2=[0.10606009252358942, 0.09247784965077381, 0.08768997835208166, 0.08588582096294924, 0.07825281574567897, 0.07471169974178306, 0.07102197487362344, 0.06980458642681989, 0.06479573706283975]
# # med_tls_rmse2=[0.23790441674417503, 0.16452901346994447, 0.1405355728153337, 0.12668903202665724, 0.10923519843043321, 0.10288789701335276, 0.09554692577326829, 0.08839737875084634, 0.0797739190329706]
# # med_ls_em_rmse2=[0.10253267020599204, 0.09209101723041144, 0.0883305704883973, 0.08753395455870115, 0.0811606306929946, 0.07695010488180598, 0.0730116240630933, 0.07121665636984083, 0.06696102396983822]
# # med_ls_rmse2=[0.14470570533690194, 0.12852904685951955, 0.12219044781309084, 0.12032952868851193, 0.11640383432470405, 0.11584932959806898, 0.11431644167898984, 0.1119528805113238, 0.10448715538906805]
# #
# # med_tls_em_rmse3=[0.11095553995399518, 0.09399346358440307, 0.08882996351062686, 0.0870357010999185, 0.08136778791175305, 0.07713542995478626, 0.07304045687017915, 0.07102129765963822, 0.06647776957267915]
# # med_tls_rmse3=[0.24192843234212072, 0.16657420938958023, 0.14266661058252983, 0.1280696346388221, 0.11044275948751277, 0.10399661240835632, 0.09642740351923593, 0.08906012393806825, 0.08042434035084253]
# # med_ls_em_rmse3=[0.1055317934246299, 0.09360649609990329, 0.08980766476495551, 0.08884505420219392, 0.08487318055300752, 0.08048569019006666, 0.0759578350940012, 0.07341911577993365, 0.06933820745540453]
# # med_ls_rmse3=[0.14584921656321012, 0.12969094273230375, 0.12316931884348342, 0.1211960386764619, 0.11729847641419834, 0.11670241876762864, 0.11521570231280788, 0.11279881763800256, 0.10534715966549481]
# #
# #
# # x_plt = train_array = np.arange(0.15, 1.0, 0.1)
# #
# # fig, axs = plt.subplots(2, 2, figsize=(12, 10))
# #
# # # 在每个子图中绘制对应的数据
# # axs[0, 0].plot(x_plt, med_tls_em_rmse0,label='TLS_EM',marker='o', linestyle='-' )
# # axs[0, 0].plot(x_plt, med_tls_rmse0,label='TLS',marker='o', linestyle='-')
# # axs[0, 0].plot(x_plt, med_ls_em_rmse0,label='LS_EM',marker='o', linestyle='-' )
# # axs[0, 0].plot(x_plt, med_ls_rmse0,label='LS',marker='o', linestyle='-' )
# # # axs[0, 0].set_xlabel('Noise')
# # # axs[0, 0].set_ylabel('RMSE')
# # # axs[0, 0].set_xticks(x_plt)
# # axs[0, 0].set_title('[1, 0.1, 0.02]')
# # axs[0, 0].locator_params(axis='x', nbins=10)  # 在横坐标上显示6个刻度
# # # axs[0, 0].title("data split:{},noise generation:{}".format(s, m))
# #
# # axs[0, 1].plot(x_plt, med_tls_em_rmse1,label='TLS_EM',marker='o', linestyle='-')
# # axs[0, 1].plot(x_plt, med_tls_rmse1, label='TLS',marker='o', linestyle='-')
# # axs[0, 1].plot(x_plt, med_ls_em_rmse1, label='LS_EM',marker='o', linestyle='-')
# # axs[0, 1].plot(x_plt, med_ls_rmse1, label='LS',marker='o', linestyle='-')
# # # axs[0, 1].set_xlabel('Noise')
# # # axs[0, 1].set_ylabel('RMSE')
# # axs[0, 1].set_title('[0.97, 0.07, 0.05]')
# # axs[0, 1].set_xticks(x_plt)
# # axs[0, 1].locator_params(axis='x', nbins=10)  # 在横坐标上显示6个刻度
# #
# #
# #
# # axs[1, 0].plot(x_plt, med_tls_em_rmse2, label='TLS_EM',marker='o', linestyle='-' )
# # axs[1, 0].plot(x_plt, med_tls_rmse2, label='TLS',marker='o', linestyle='-')
# # axs[1, 0].plot(x_plt, med_ls_em_rmse2, label='LS_EM',marker='o', linestyle='-' )
# # axs[1, 0].plot(x_plt, med_ls_rmse2, label='LS',marker='o', linestyle='-')
# # axs[1, 0].set_title('[0.99, 0.04, 0.03]')
# # axs[1, 0].set_xticks(x_plt)
# # axs[1, 0].locator_params(axis='x', nbins=10)  # 在横坐标上显示6个刻度
# #
# # axs[1, 1].plot(x_plt, med_tls_em_rmse3, label='TLS_EM',marker='o', linestyle='-' )
# # axs[1, 1].plot(x_plt, med_tls_rmse3, label='TLS',marker='o', linestyle='-')
# # axs[1, 1].plot(x_plt, med_ls_em_rmse3, label='LS_EM' ,marker='o', linestyle='-')
# # axs[1, 1].plot(x_plt, med_ls_rmse3, label='LS',marker='o', linestyle='-')
# # # axs[1, 1].set_xlabel('Noise')
# # # axs[1, 1].set_ylabel('RMSE')
# # axs[1, 1].set_title('[1,0.06,0.08]')
# # axs[1, 1].set_xticks(x_plt)
# # axs[1, 1].locator_params(axis='x', nbins=10)  # 在横坐标上显示6个刻度
# # # 添加标题和标签
# # # fig.suptitle('Comparison of RMSE for Four Methods')
# # for ax in axs.flat:
# #     ax.set(xlabel='train_size', ylabel='RMSE')
# #     ax.legend()
# #
# #
# #
# #
# #
# #
# #
# # #
# # # x_plt = np.arange(0.25, 1.01, 0.05)
# # #     # x_plt=[0.25,0.3]
# # # print(x_plt)
# # # plt.plot(x_plt,med_tls_em_rmse, )
# # # plt.plot(x_plt,med_tls_rmse)
# # # plt.plot(x_plt,med_ls_em_rmse, )
# # # plt.plot(x_plt,med_ls_rmse)
# # # plt.legend(['TLS_EM', 'TLS','LS_EM', 'LS'])  #
# # # plt.xlabel('Noise')
# # # plt.ylabel('RMSE')
# # #
# # # plt.title("data split:50,noise generation:20")
# # # plt.show()
# #
# # #
# # # # 绘制折线图
# # # plt.figure(figsize=(10, 6))
# # # plt.plot(x_plt, med_tls_em_rmse, label='Med TLS EM RMSE', marker='o')
# # # plt.plot(x_plt, med_tls_rmse, label='Med TLS RMSE', marker='o')
# # # plt.plot(x_plt, med_ls_em_rmse, label='Med LS EM RMSE', marker='o')
# # # plt.plot(x_plt, med_ls_rmse, label='Med LS RMSE', marker='o')
# # #
# # # # 添加标题和标签
# # # plt.title('Comparison of RMSE for Four Methods')
# # # plt.xlabel('X-axis Label')
# # # plt.ylabel('RMSE Values')
# # #
# # # # 添加图例
# # # plt.legend()
# #
# # # 显示图形
# # plt.show()
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

fig, axs = plt.subplots(2, 4, figsize=(30, 6))

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
axins.plot(np.arange(0,3,1), med_tls_em_rmse0[9:12],label='TLS_EM',linestyle='-',marker='^',color='blue' )
axins.plot(np.arange(0,3,1), med_ls_em_rmse0[9:12],label='LS_EM',linestyle='-',marker='*',color='green' )
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
axins.plot(np.arange(0,3,1), med_tls_em_rmse1[9:12],label='TLS_EM',linestyle='-',marker='^',color='blue' )
axins.plot(np.arange(0,3,1), med_ls_em_rmse1[9:12],label='LS_EM',linestyle='-',marker='*',color='green' )
# 不显示放大区域的坐标刻度
axins.set_xticks([])
axins.set_yticks([])



axs[0, 2].plot(x_plt, med_tls_em_rmse2, label='TLS_EM',marker='^', linestyle='-' )
axs[0, 2].plot(x_plt, med_tls_rmse2, label='TLS',marker='D', linestyle='-')
axs[0, 2].plot(x_plt, med_ls_em_rmse2, label='LS_EM',marker='*', linestyle='-' )
axs[0, 2].plot(x_plt, med_ls_rmse2, label='LS',marker='o', linestyle='-')
axs[0, 2].set_title('[0.99, 0.04, 0.03]')
axs[0, 2].set_xticks(x_plt)
axs[0, 2].locator_params(axis='x', nbins=10)  # 在横坐标上显示6个刻度
axins = inset_axes(axs[0, 2], width="20%", height="20%", loc='lower right',bbox_to_anchor=(0.01, 0.07, 1, 1),
                   bbox_transform=axs[0,2].transAxes)
# 在放大区域中绘制数据
axins.plot(np.arange(0,3,1), med_tls_em_rmse2[9:12],label='TLS_EM',linestyle='-',marker='^',color='blue' )
axins.plot(np.arange(0,3,1), med_ls_em_rmse2[9:12],label='LS_EM',linestyle='-',marker='*',color='green' )
# 不显示放大区域的坐标刻度
axins.set_xticks([])
axins.set_yticks([])


axs[0, 3].plot(x_plt, med_tls_em_rmse3, label='TLS_EM',marker='^', linestyle='-' )
axs[0, 3].plot(x_plt, med_tls_rmse3, label='TLS',marker='D', linestyle='-')
axs[0, 3].plot(x_plt, med_ls_em_rmse3, label='LS_EM' ,marker='*', linestyle='-')
axs[0, 3].plot(x_plt, med_ls_rmse3, label='LS',marker='o', linestyle='-')
# axs[1, 1].set_xlabel('Noise')
# axs[1, 1].set_ylabel('RMSE')
axs[0, 3].set_title('[1,0.06,0.08]')
axs[0, 3].set_xticks(x_plt)
axs[0, 3].locator_params(axis='x', nbins=10)  # 在横坐标上显示6个刻度
axins = inset_axes(axs[0, 3], width="20%", height="20%", loc='lower right',bbox_to_anchor=(0.01, 0.07, 1, 1),
                   bbox_transform=axs[0,3].transAxes)
# 在放大区域中绘制数据
axins.plot(np.arange(0,3,1), med_tls_em_rmse3[9:12],label='TLS_EM',linestyle='-',marker='^',color='blue' )
axins.plot(np.arange(0,3,1), med_ls_em_rmse3[9:12],label='LS_EM',linestyle='-',marker='*',color='green' )
# 不显示放大区域的坐标刻度
axins.set_xticks([])
axins.set_yticks([])

# 添加标题和标签
plt.subplots_adjust(left=0.07, right=0.95, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)
# plt.subplots_adjust(wspace=0.5, hspace=0.5)
# 在整个图上添加 'a', 'b', 'c', 'd' 的标注
fig.text(0.03, 0.9, 'a', fontsize=14, fontweight='bold')
fig.text(0.27, 0.9, 'b', fontsize=14, fontweight='bold')
fig.text(0.5, 0.9, 'c', fontsize=14, fontweight='bold')
fig.text(0.73, 0.9, 'd', fontsize=14, fontweight='bold')
fig.text(0.03, 0.46, 'e', fontsize=14, fontweight='bold')
fig.text(0.27, 0.46, 'f', fontsize=14, fontweight='bold')
fig.text(0.5, 0.46, 'g', fontsize=14, fontweight='bold')
fig.text(0.73, 0.46, 'h', fontsize=14, fontweight='bold')

# fig.suptitle('Comparison of RMSE for Four Methods')
for ax in axs.flat:
    ax.set(xlabel='noise level', ylabel='RMSE')
    ax.legend()

plt.show()
