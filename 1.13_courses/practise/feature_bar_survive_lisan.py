import sys
import io
import pandas as pd 
import matplotlib.pyplot as plt 

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding="utf8")

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
data_train = pd.read_csv("D:\\CodeForFun\\machinelearning\\1.13 courses\\titanic\\data\\train.csv")


fig = plt.figure()
fig.set(alpha=0.2)
Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived ==1].value_counts()
# print(Survived_0)
# print(Survived_1)

# print(data_train.columns)


df = pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar',stacked = True)
plt.title(u'各乘客等级获救情况')
plt.xlabel(u'乘客等级')
plt.ylabel(u'人数')

plt.show()




