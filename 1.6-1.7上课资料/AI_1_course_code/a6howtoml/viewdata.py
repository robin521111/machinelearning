# -*- coding: utf-8 -*-
import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
#data_train = pd.read_csv("Train.csv")
data_train = pd.read_csv("D:\\project\\peixun\\ai_course_project_px\\1_intro\\4_anli_project_titanic\\Kaggle_Titanic_Chinese\\Kaggle_Titanic-master\\train.csv")
print(data_train.columns)
#data_train[data_train.Cabin.notnull()]['Survived'].value_counts()

# (2)
print(data_train.info())

# (3)
print(data_train.describe())
