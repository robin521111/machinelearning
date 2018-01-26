import pandas as pd 
import numpy as np 
data_train = pd.read_csv("C:\\Codeforfun\\machinelearning\\1.21_homework\\code\\7_titanic\\data\\train.csv")

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

#Label Encoder Method
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data2 = data_train["Embarked"]

data2 = data_train["Embarked"].astype('category')
# print(data2)

encoder_data2 = data2.cat.codes
print(encoder_data2.head())


le.fit(data2)
print(data2)
# print(data_train.dtypes)
# print(data_train[:,-1].value_counts())
# le.fit(data_value)
# leed_list= list(le.fit(data1))
# print(leed_list)