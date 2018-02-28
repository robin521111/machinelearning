# -*- coding: utf-8 -*-
# 加载相关模块和库
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
import time
import sys
import io
#改变标准输出的默认编码
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
def run_main():
    """
    以前说过：随机森林是我们的第一选择，但是目前来看效果可能并不好
    这就是调参的原因
    """
    def cal_acc(a, b):
        return len(np.where(a==b)[0])*100/len(b)

    datas = np.load("data/fraud_balanced.npz")
    X_train=datas['X_train']
    X_test=datas['X_test']
    y_train=datas['y_train']
    y_test=datas['y_test']
    models = {"逻辑回归":LogisticRegression(), 
             "随机森林1":RandomForestClassifier(criterion='gini', max_depth=2, n_estimators=5),
             "随机森林2":RandomForestClassifier(criterion='gini', max_depth=6, n_estimators=20)}
    for name in models:
        model = models[name]
        model.fit(X_train, y_train)
        st = time.clock()
        for itr in range(100):
            y_pred = model.predict(X_test)
        ed = time.clock()
        y_pred_train = model.predict(X_train)
        print("Method:{}; Train acc:{}; Test acc:{}; 1000iter time consumption:{}".format(name, cal_acc(y_test, y_pred), cal_acc(y_pred_train, y_train), ed-st))
    


if __name__ == '__main__':
    run_main()



