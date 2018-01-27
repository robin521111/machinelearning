#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
独立成分分析
"""
print(__doc__)

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.decomposition import PCA
mpl.style.use('fivethirtyeight')


mean = [0, 0]   # 平均值
cov = [[1, 0.9], [0.9, 1]]   # 协方差
x, y = np.random.multivariate_normal(mean, cov, 1000).T
x = np.reshape(x, [-1, 1])
y = np.reshape(y, [-1, 1])
X = np.concatenate([x, y], axis=1)
label = np.zeros_like(x[:,0])
label[x[:,0]>1]=1
label[x[:,0]<-1]=1
pca = PCA(n_components=0.99)
X_pca = pca.fit_transform(X)

fig = plt.figure(1)
ax = fig.add_subplot(211)
ax.scatter(X[:, 0], X[:, 1], c=label)
ax.axis("equal")
ax = fig.add_subplot(212)
ax.scatter(X_pca[:, 0]*pca.explained_variance_[0], X_pca[:, 1]*pca.explained_variance_[1], c=label)
ax.axis("equal")
plt.show()