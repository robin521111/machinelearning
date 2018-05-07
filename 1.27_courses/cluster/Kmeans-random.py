from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.animation as animation
#调整图片风格
mpl.style.use('fivethirtyeight')
#定义xy网格，用于绘制等值线图

X = np.random.random([1000, 2])
fig = plt.figure()
ax = fig.add_subplot(111)
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
#预测可能性
clt = KMeans(n_clusters=3, random_state=6)
clt.fit(X)
Z = clt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
yp = clt.predict(X)
ax.contourf(xx, yy, Z, alpha=.8)
#绘制散点图
ax.scatter(X[:, 0], X[:, 1], c=yp, edgecolors='k')

def update(num, img, X, xx, yy):
    #X = np.random.random([1000, 2])
    clt = KMeans(n_clusters=3, random_state=num*10)
    clt.fit(X)
    Z = clt.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    yp = clt.predict(X)
    img.cla()
    img.text(0.0, 0.6, "p=%f"%num)
    img.contourf(xx, yy, Z, alpha=.8)
    #绘制散点图
    img.scatter(X[:, 0], X[:, 1], c=yp, edgecolors='k')

line_ani = animation.FuncAnimation(fig, update, 200000, fargs=(ax, X, xx, yy),
                                   interval=50, blit=False)
plt.axis("equal")
plt.show()