# -*- coding: utf-8 -*-
import pandas as pd #数据分析
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
data_train = pd.read_csv(
    "/Users/robin/Documents/MachineLearning/machinelearning/1.21_courses/4_decisiontree/6_titanic/Kaggle_Titanic-master/train.csv")

#看看各性别的获救情况
Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df=pd.DataFrame({u'男性':Survived_m, u'女性':Survived_f})
df.plot(kind='bar', stacked=True)
plt.title(u"按性别看获救情况")
plt.xlabel(u"性别")
plt.ylabel(u"人数")
plt.show()

