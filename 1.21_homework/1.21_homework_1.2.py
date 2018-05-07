import pandas as pd 
import numpy as np 
from  sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score

data_train = pd.read_csv("C:\\Codeforfun\\machinelearning\\1.21_homework\\code\\7_titanic\\data\\train.csv")
print(data_train.head)


# onehot = OneHotEncoder()



# onehot.fit(data_train)
