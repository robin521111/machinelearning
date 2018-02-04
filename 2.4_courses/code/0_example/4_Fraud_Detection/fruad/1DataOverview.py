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
import sys
import io
#改变标准输出的默认编码
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

def run_main():
    """
    数据概览
    """
    csvfile_path = 'data/PS_20174392719_1491204439457_log.csv'
    raw_data = pd.read_csv(csvfile_path).sample(n=10000)
    # 查看数据集信息
    print('数据预览：')
    print(raw_data.head())

    print('数据统计信息：')
    print(raw_data.describe())

    print('数据集基本信息：')
    print(raw_data.info())

    # 2. 探索性数据分析(EDA)及可视化
    # 查看转账类型
    print('转账类型记录统计：')
    print(raw_data['type'].value_counts())

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    raw_data['type'].value_counts().plot(kind='bar', title='Transaction Type', ax=ax, figsize=(8, 4))
    plt.show()

    # 查看转账类型和欺诈标记的记录
    ax = raw_data.groupby(['type', 'isFraud']).size().plot(kind='bar')
    ax.set_title('# of transactions vs (type + isFraud)')
    ax.set_xlabel('(type, isFraud)')
    ax.set_ylabel('# of transaction')

    # 添加标注
    for p in ax.patches:
        ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))

    # 查看转账类型和商业模型标记的欺诈记录
    ax = raw_data.groupby(['type', 'isFlaggedFraud']).size().plot(kind='bar')
    ax.set_title('# of transactions vs (type + isFlaggedFraud)')
    ax.set_xlabel('(type, isFlaggedFraud)')
    ax.set_ylabel('# of transaction')
    used_data = raw_data[(raw_data['type'] == 'TRANSFER') | (raw_data['type'] == 'CASH_OUT')]
    # 丢掉不用的数据列
    used_data.drop(['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, inplace=True)
    # 重新设置索引
    used_data = used_data.reset_index(drop=True)

    # 将type转换成类别数据，即0, 1
    type_label_encoder = preprocessing.LabelEncoder()
    type_category = type_label_encoder.fit_transform(used_data['type'].values)
    used_data['typeCategory'] = type_category

    print(used_data.head())

    # 查看变量间的相关性
    sns.heatmap(used_data.corr())

    # 查看转账类型记录个数
    ax = used_data['type'].value_counts().plot(kind='bar', title="Transaction Type", figsize=(6,6))
    for p in ax.patches:
        ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))
    plt.show()

    # 查看转账类型中欺诈记录个数
    ax = pd.value_counts(used_data['isFraud'], sort=True).sort_index().plot(kind='bar', title="Fraud Transaction Count")
    for p in ax.patches:
        ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()))
    plt.show()

    # 准备数据
    feature_names = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'typeCategory']
    X = used_data[feature_names]
    y = used_data['isFraud']
    print(X.head())
    print(y.head())

    # 处理不平衡数据
    # 欺诈记录的条数
    number_records_fraud = len(used_data[used_data['isFraud'] == 1])
    # 欺诈记录的索引
    fraud_indices = used_data[used_data['isFraud'] == 1].index.values

    # 得到非欺诈记录的索引
    nonfraud_indices = used_data[used_data['isFraud'] == 0].index

    # 随机选取相同数量的非欺诈记录
    random_nonfraud_indices = np.random.choice(nonfraud_indices, number_records_fraud, replace=False)
    random_nonfraud_indices = np.array(random_nonfraud_indices)

    # 整合两类样本的索引
    under_sample_indices = np.concatenate([fraud_indices, random_nonfraud_indices])
    under_sample_data = used_data.iloc[under_sample_indices, :]
                          
    X_undersample = under_sample_data[feature_names].values
    y_undersample = under_sample_data['isFraud'].values

    # 显示样本比例
    print("非欺诈记录比例: ", len(under_sample_data[under_sample_data['isFraud'] == 0]) / len(under_sample_data))
    print("欺诈记录比例: ", len(under_sample_data[under_sample_data['isFraud'] == 1]) / len(under_sample_data))
    print("欠采样记录数: ", len(under_sample_data))


if __name__ == '__main__':
    run_main()



