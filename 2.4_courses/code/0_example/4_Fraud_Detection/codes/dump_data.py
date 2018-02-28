# -*- coding: utf-8 -*-
import pandas as pd #数据分析
# data_train = pd.read_csv("D:\\project\\peixun\\ai_course_project_px\\1_intro\\4_anli_project_titanic\\Kaggle_Titanic_Chinese\\Kaggle_Titanic-master\\train.csv")
# # (1) 看列名
# print(data_train.columns)

# # (2) 看每列性质，空值和类型
# print(data_train.info())

# # (3) 看每列统计信息
# print(data_train.describe())
# df.to_csv('/tmp/9.csv',columns=['open','high'],index=False,header=False)
# 不要列头，不要索引，只要open,high两列。

data_train = pd.read_csv("D:\\PS_20174392719_1491204439457_log.csv")
print(data_train.columns)
sample_data = data_train.sample(10000)
sample_data.to_csv("D:\\dataset\\ai_course_data\\fraud_detection\\sample\\fraud_detection_sample.csv", index=False)

