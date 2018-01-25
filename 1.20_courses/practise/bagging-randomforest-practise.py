print(__doc__)

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons,make_circles,make_classification

X, y = make_moons(noise=.1,random_state=1)

rdf = RandomForestClassifier()

rdf.fit(X,y)

import matplotlib.pyplot as plt 
import matplotlib as mpl 
import numpy as np 

#调整图片风格
mpl.style.use('fivethirtyeight')
#定义xy网格，用于绘制等值线图
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
#预测可能性
Z = rdf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.8)
#绘制散点图
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title("RandomForest")
plt.axis("equal")
plt.show()
