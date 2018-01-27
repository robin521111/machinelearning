#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
马氏距离与样本分布有关，用的比较少。
可以看成空间仿射变换后距离
"""
print(__doc__)


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.animation as animation
import scipy.spatial.distance as dst
mpl.style.use('fivethirtyeight')
fig = plt.figure()
ax = fig.add_subplot(111)
mean = [0, 0]      # 平均值
cov = [[2, 1], [1, 2]]   # 协方差
x, y = np.random.multivariate_normal(mean, cov, 1000).T
cov_mat = np.matrix(np.cov(x, y))

#定义xy网格，用于绘制等值线图
x_min, x_max = - 5,  + 5
y_min, y_max = - 5,  + 5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

p = np.c_[xx.ravel(), yy.ravel()]
Z = []
for itr in p:
    Z.append(dst.mahalanobis([0, 0], itr, cov_mat.I))
Z = np.reshape(Z, xx.shape)
surf = ax.contourf(xx, yy, Z)
ax.scatter(x, y, alpha=0.6)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
plt.axis("equal")
plt.show()
