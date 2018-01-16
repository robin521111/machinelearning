import pandas as pd 
import sys
import io

sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

data_train= pd.read_csv("D:\\CodeForFun\\machinelearning\\1.13 courses\\titanic\\data\\train.csv")

print("看列名",data_train.columns)

print("看每列性质,空值和类型",data_train.info())

print(data_train.describe())


import matplotlib.pyplot as plt 
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


plt.subplot2grid((2,3),(1,0),colspan=2)
data_train.Age[data_train.Pclass==1].plot(kind='kde')
data_train.Age[data_train.Pclass==2].plot(kind='kde')
data_train.Age[data_train.Pclass==3].plot(kind='kde')
plt.xlabel('年龄')
plt.ylabel('密度')
plt.title("各等级的乘客年龄分布")
plt.legend(("头等舱","2等舱","3等舱"),loc="best")
plt.show()
