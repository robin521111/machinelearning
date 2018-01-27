#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
谱聚类
"""
print(__doc__)

from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.cluster import SpectralClustering
import numpy as np
import matplotlib.pyplot as plt

centers = [[0, 1], [-1, -1], [1, -1]]
X, y = make_blobs(n_samples=1500, random_state=170)
trs = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X = np.dot(X, trs)    
clt = SpectralClustering(n_clusters=3)
yp = clt.fit_predict(X)


plt.scatter(X[:, 0], X[:, 1], c=yp, edgecolors='k')
plt.axis("equal")
plt.show()