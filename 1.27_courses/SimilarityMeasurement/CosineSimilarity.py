#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
余弦相似度与向量的幅值无关，只与向量的方向相关，
在文档相似度（TF-IDF）和图片相似性（histogram）计算上都有它的身影。
需要注意一点的是，余弦相似度受到向量的平移影响，
上式如果将 x 平移到 x+1, 余弦值就会改变。
怎样才能实现平移不变性？这就是下面要说的皮尔逊相关系数（Pearson correlation）
"""
print(__doc__)
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.animation as animation
#调整图片风格
mpl.style.use('fivethirtyeight')
#定义xy网格，用于绘制等值线图
x_min, x_max = - 5,  + 5
y_min, y_max = - 5,  + 5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.100000001),
                     np.arange(y_min, y_max, 0.100000001))
# 

fig = plt.figure()
ax = fig.add_subplot(111)
p = np.c_[xx.ravel(), yy.ravel()]
p2 = np.square(p)
v2 = 1/np.sqrt(np.sum(p2, axis=1))
d = np.array([[2.], [1.]])
Z = np.dot(p, d)
Z = Z*v2.reshape(Z.shape)/np.sqrt(2)
print(np.shape(v2))
Z = Z.reshape(xx.shape)
surf = ax.contourf(xx, yy, Z)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

plt.axis("equal")
plt.show()
