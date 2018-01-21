# -*- coding: utf-8 -*-
import pandas as pd #数据分析
data_train = pd.read_csv(
    "/Users/robin/Documents/MachineLearning/machinelearning/1.21_courses/4_decisiontree/6_titanic/Kaggle_Titanic-master/train.csv")
# (1) 看列名
print(data_train.columns)

# (2) 看每列性质，空值和类型
print(data_train.info())

# (3) 看每列统计信息
print(data_train.describe())
