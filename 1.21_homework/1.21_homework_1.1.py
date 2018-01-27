import pandas as pd 
import numpy as np 
data_train = pd.read_csv("D:\\CodeForFun\\machinelearning\\1.21_homework\\code\\7_titanic\\data\\train.csv")

# print(data_train.info())
# print(data_train.describe)

data_value = data_train.values

data1 = data_value[:,-1]
# print(data1)

attr = set(data1)
dic = dict()

for idx, itr in enumerate(attr):
    dic[itr] = idx
n_data = []
for itr in data1:
    n_data.append(dic[itr])

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

# le.fit(attr)


# leed_list= list(le.fit(data1))
# print(leed_list)

