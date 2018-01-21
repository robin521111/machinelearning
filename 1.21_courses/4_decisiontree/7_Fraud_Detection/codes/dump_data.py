# -*- coding: utf-8 -*-
# 加载相关模块和库
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import zipfile
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn import ensemble

def run_main():
    """
        主函数
    """
    # 1. 准备阶段
    # 声明变量
    dataset_path = 'D:\\data\\fraud_detection\\PS_20174392719_1491204439457_log.csv'
    #zipfile_path = os.path.join(dataset_path, 'PS_20174392719_1491204439457_log.csv.zip')
    csvfile_path = os.path.join(dataset_path, 'PS_20174392719_1491204439457_log.csv')

    # 解压数据集
    # with zipfile.ZipFile(zipfile_path) as zf:
    #     zf.extractall(dataset_path)
    
    # 读取数据集
    sample = 10000
    raw_data = pd.read_csv(csvfile_path).sample(n=sample)
    raw_data.to_csv("D:\\data\\fraud_detection\\" + sample + ".csv")