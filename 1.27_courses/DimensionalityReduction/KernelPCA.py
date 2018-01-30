#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
将空间进行变换后进行降维。
"""
print(__doc__)

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.decomposition import KernelPCA
mpl.style.use('fivethirtyeight')
from sklearn.datasets import make_circles

np.random.seed(0)

X, y = make_circles(n_samples=400, factor=.2, noise=.01)

pca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
X_pca = pca.fit(X).fit_transform(X)

fig = plt.figure()
ax = fig.add_subplot(211)
ax.scatter(X[:, 0], X[:, 1], c=y)
ax.axis("equal")
ax = fig.add_subplot(212)
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
ax.axis("equal")
plt.show()