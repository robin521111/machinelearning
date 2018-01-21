# -*- coding: utf-8 -*-
import pandas as pd #数据分析
data_train = pd.read_csv("D:\\project\\peixun\\ai_course_project_px\\1_intro\\4_anli_project_titanic\\Kaggle_Titanic_Chinese\\Kaggle_Titanic-master\\train.csv")
# (1) 看列名
print(data_train.columns)

# (2) 看每列性质，空值和类型
print(data_train.info())

# (3) 看每列统计信息
print(data_train.describe())
