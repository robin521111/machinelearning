import sys
import io
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')


import pandas as pd
import matplotlib.pyplot as plt 

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

data_train = pd.read_csv("D:\\CodeForFun\\machinelearning\\1.13 courses\\titanic\\data\\train.csv")

print(data_train.columns)

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
survived_1 =  data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({u'获救':survived_1,u'未获救':Survived_0})
df.plot(kind='bar',stacked=True)
plt.title(u"各乘客等级的获救情况")
plt.xlabel(u"乘客等级")
plt.ylabel(u"人数")
plt.show()
