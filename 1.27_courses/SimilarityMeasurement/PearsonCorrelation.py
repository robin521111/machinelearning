#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
皮尔逊相关系数具有平移不变性和尺度不变性
计算出了两个向量（维度）的相关性。
不过，一般我们在谈论相关系数的时候，
将 x 与 y 对应位置的两个数值看作一个样本点，
皮尔逊系数用来表示这些样本点分布的相关性。 
"""
print(__doc__)
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.animation as animation
mpl.style.use('fivethirtyeight')

mean = [1, 3]      # 平均值
cov = [[1, 0.9], [0.9, 1]]   # 协方差
x1, y1 = np.random.multivariate_normal(mean, cov, 1000).T
x = x1 - np.mean(x1)
y = y1 - np.mean(y1)
corr = np.sum(x*y)/np.sqrt(np.sum(x*x))/np.sqrt(np.sum(y*y))

plt.scatter(x1, y1)
plt.text(1, 3, "P=%f"%corr)
plt.show()