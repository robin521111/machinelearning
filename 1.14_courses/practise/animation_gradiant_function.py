# -*- coding: utf-8 -*-
# cangye@Hotmail.com
"""
动态展示梯度下降过程，代码不需要理解
"""
print(__doc__)


import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib.animation as animation
import numpy as np

def function(x, y):
    return x**2+y**2
def grad1(x, y):
    return [-2*x, -2*y]

def update(num, data, lines, scate):
    # 给定不同梯度参数
    eta=[1, 0.5, 0.1, 0.01, 0.001]
    rand=[0, 0, 0, 0, 5]
    for itr in range(5):
        data[itr][0].append(data[itr][0][-1] + eta[itr] * grad1(data[itr][0][-1], data[itr][1][-1])[0] * (rand[itr] * (np.random.random()-0.5) + 1))
        data[itr][1].append(data[itr][1][-1] + eta[itr] * grad1(data[itr][0][-1], data[itr][1][-1])[1] * (rand[itr] * (np.random.random()-0.5) + 1))
        lines[itr].set_xdata(data[itr][0])
        lines[itr].set_ydata(data[itr][1])
        z = function(np.array(data[itr][0]), np.array(data[itr][1]))
        lines[itr].set_3d_properties(z)
        scate[itr].set_ydata(data[itr][1][-1])
        scate[itr].set_xdata(data[itr][0][-1])
        scate[itr].set_3d_properties(z[-1])
    
# 调整图片风格
mpl.style.use('seaborn-darkgrid')

# 绘制三维曲面
x = y = np.arange(-3.0, 3.0, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array([function(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.axis('equal')
ax.plot_surface(X, Y, Z, alpha=0.5)

lines=[ax.plot([],[],[])[0] for itr in range(5)]
scate=[ax.plot([],[],[],'o-')[0] for itr in range(5)]
data=[[[2.5],[2.5]],[[2.5],[2.5]],[[2.5],[2.5]],[[2.5],[2.5]],[[2.5],[2.5]]]

line_ani = animation.FuncAnimation(fig, update, 100, fargs=(data, lines, scate),
                                   interval=50, blit=False)
plt.show()