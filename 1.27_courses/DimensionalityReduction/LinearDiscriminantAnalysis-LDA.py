#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
使得空间中两类的距离尽可能的远。
"""
print(__doc__)


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

mean = [0, 0]   # 平均值
cov = [[1, 0.9], [0.9, 1]]   # 协方差
x, y = np.random.multivariate_normal(mean, cov, 1000).T
x = np.reshape(x, [-1, 1])
y = np.reshape(y, [-1, 1])
X = np.concatenate([x, y], axis=1)
label = np.zeros_like(x[:,0])
label[x[:,0]>0]=1
pca = LinearDiscriminantAnalysis()
X_pca = pca.fit_transform(X, label)

fig = plt.figure()
ax = fig.add_subplot(211)
ax.scatter(X[:, 0], X[:, 1], c=label)
ax.axis("equal")
ax = fig.add_subplot(212)
ax.scatter(X_pca, np.zeros_like(X_pca), c=label)
ax.axis("equal")
plt.show()
