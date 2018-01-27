import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib.animation as animation
import numpy as np
from matplotlib.colors import LogNorm
from sklearn import mixture
from sklearn.datasets import make_moons, make_circles, make_blobs
mpl.style.use('fivethirtyeight')
n_samples = 300

centers = [[0, 1], [-1, -1], [1, -1]]
X, y = make_blobs(n_samples=1500, random_state=170)
trs = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X = np.dot(X, trs)   

clf = mixture.GaussianMixture(max_iter=300, n_components=3, covariance_type='full')
clf.fit(X)

# display predicted scores by the model as a contour plot
x_min, x_max = X[:, 0].min() - .2, X[:, 0].max() + .2
y_min, y_max = X[:, 1].min() - .2, X[:, 1].max() + .2
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

Z = clf.score_samples(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

YY = clf.predict(X)
YY = np.array(YY)
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_surface(xx, yy, Z, alpha=0.5)

ax.scatter(X[:, 0], X[:, 1], c=YY)

plt.title('GMM')
plt.axis('tight')
plt.show()

