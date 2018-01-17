# -*- coding: utf-8 -*-

# 这个ipython notebook主要是我解决Kaggle Titanic问题的思路和过程
# (1)
import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
import sys
#reload(sys)
#sys.setdefaultencoding( "utf-8" )
import matplotlib.pyplot as plt
#data_train = pd.read_csv("Train.csv")
data_train = pd.read_csv("D:\\project\\peixun\\ai_course_project_px\\1_intro\\4_anli_project_titanic\\Kaggle_Titanic_Chinese\\Kaggle_Titanic-master\\train.csv")
print(data_train.columns)

# (5)
#看看各乘客等级的获救情况
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各乘客等级的获救情况")
plt.xlabel(u"乘客等级")
plt.ylabel(u"人数")
plt.show()
