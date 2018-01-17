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

#看看各性别的获救情况
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df=pd.DataFrame({u'男性':Survived_m, u'女性':Survived_f})
df.plot(kind='bar', stacked=True)
plt.title(u"按性别看获救情况")
plt.xlabel(u"性别")
plt.ylabel(u"人数")
plt.show()
