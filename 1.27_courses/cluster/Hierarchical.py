#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
层次聚类
自低向上，初始中，每个点作为一类。
"""
print(__doc__)

from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt

centers = [[0, 1], [-1, -1], [1, -1]]
X, y = make_blobs(n_samples=1500, random_state=170)
trs = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X = np.dot(X, trs)    
clt = AgglomerativeClustering(linkage="ward")
yp = clt.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=yp, edgecolors='k')
plt.axis("equal")
plt.show()