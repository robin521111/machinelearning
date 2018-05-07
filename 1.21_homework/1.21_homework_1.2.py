import pandas as pd 
import numpy as np 
from  sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score

data_train = pd.read_csv("C:\\Codeforfun\\machinelearning\\1.21_homework\\code\\7_titanic\\data\\train.csv")
print(data_train.head)


<<<<<<< HEAD
=======
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
data3 = data_train["Embarked"]

print(type(data3))
enc.fit([data3,[1,2,1]])


>>>>>>> 9c7cd770a298a938eca3a00ae7935f0b6101c792
# onehot = OneHotEncoder()
# onehot.fit(data_train)
