import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print(__doc__)

import pandas as pd 
data_train = pd.read_csv("D:\\CodeForFun\\machinelearning\\1.13 courses\\titanic\\data\\train.csv")

data = data_train.values

# print(data_train)
print(data)
data1 = data[:,-1]
print(data1)

attr = set(data1)
print(set(attr))


dic = dict()

for idx,itr in enumerate(attr):
    dic[itr]=idx
n_data1  = []
for itr in data1:
    n_data1.append(dic[itr])
print("原如字符串数据：", data1)
print("转换后数据：",n_data1)
