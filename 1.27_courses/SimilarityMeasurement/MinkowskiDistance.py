#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
闵可夫斯基距离
看动态图理解。
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
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
# 

fig = plt.figure()
ax = fig.add_subplot(111)
p = np.c_[xx.ravel(), yy.ravel()]
d = np.array([0., 0.])
Z = np.sum(np.power(np.abs(p-d), 0.4), axis=1)
Z = Z.reshape(xx.shape)
surf = ax.contourf(xx, yy, Z)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
def update(num, img, xx, yy):
    num = num%2000
    num = (num+1)/100
    p = np.c_[xx.ravel(), yy.ravel()]
    d = np.array([0., 0.])
    Z = np.sum(np.power(np.abs(p-d), num), axis=1)
    Z = Z.reshape(xx.shape)
    img.cla()
    img.text(-3, 3, "p=%f"%num)
    img.contourf(xx, yy, Z)
    img.set_xlim(-5, 5)
    img.set_ylim(-5, 5)

line_ani = animation.FuncAnimation(fig, update, 200000, fargs=(ax, xx, yy),
                                   interval=50, blit=False)

plt.axis("equal")
plt.show()
