#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
Kmeans算法
最常用的聚类方法
ref：zhang
"""
print(__doc__)

import matplotlib.pyplot as plt
from scipy.linalg import norm
import numpy.matlib as ml
import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs
np.random.seed(0)
def kmeans(X, k, observer=None, threshold=1e-15, maxiter=300, style="kmeans"):
    N = len(X)
    labels = np.zeros(N, dtype=int)
    
    centers = X[np.random.choice(len(X), k)]
    itr = 0


    def calc_J():
        """
        计算所有点距离和
        """
        sums = 0
        for i in range(N):
            sums += norm(X[i]-centers[labels[i]])
        return sums
 
    def distmat(X, Y):
        """
        计算距离
        """
        n = len(X)
        m = len(Y)
        xx = ml.sum(X*X, axis=1)
        yy = ml.sum(Y*Y, axis=1)
        xy = ml.dot(X, Y.T)
        return np.tile(xx, (m, 1)).T+np.tile(yy, (n, 1)) - 2*xy
 
    Jprev = calc_J()
    while True:
        #绘图
        observer(itr, labels, centers)

        dist = distmat(X, centers)
        labels = dist.argmin(axis=1)
        #再次绘图
        observer(itr, labels, centers)
        # 重新计算聚类中心
        if style=="kmeans":
            for j in range(k):
                idx_j = (labels == j).nonzero()
                centers[j] = X[idx_j].mean(axis=0)
        elif style=="kmedoids":
            for j in range(k):
                idx_j = (labels == j).nonzero()
                distj = distmat(X[idx_j], X[idx_j])
                distsum = ml.sum(distj, axis=1)
                icenter = distsum.argmin()
                centers[j] = X[idx_j[0][icenter]]

        J = calc_J()
        itr += 1
 
        if Jprev-J < threshold:
            """
            当中心不再变化停止迭代
            """
            break
        Jprev = J
        if itr >= maxiter:
            break

        
 
if __name__ == '__main__':
    # 加载数据点
    X, _ = make_blobs(n_samples=1500, random_state=170)
 
    def observer(iter, labels, centers):
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
 
        # 绘制数据点
        data_colors=[colors[lbl] for lbl in labels]
        plt.scatter(X[:, 0], X[:, 1], c=data_colors, s=100, alpha=0.2, marker="o")
        # 绘制中心
        plt.scatter(centers[:, 0], centers[:, 1], s=200, c='k', marker="^")
 
        plt.show()
 
    kmeans(X, 3, observer=observer)