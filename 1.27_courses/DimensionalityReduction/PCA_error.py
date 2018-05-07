import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.decomposition import PCA
mpl.style.use('fivethirtyeight')
from sklearn.datasets import make_circles

np.random.seed(0)

X, y = make_circles(n_samples=400, factor=.3, noise=.05)

pca = PCA(n_components=0.99)
X_pca = pca.fit_transform(X)

print(pca.explained_variance_)
fig = plt.figure()
ax = fig.add_subplot(211)
ax.scatter(X[:, 0], X[:, 1], c=y)
ax.axis("equal")
ax = fig.add_subplot(212)
ax.scatter(X_pca[:, 0]*pca.explained_variance_[0], X_pca[:, 1]*pca.explained_variance_[1], c=y)

ax.axis("equal")
plt.show()