import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib.animation as animation
import numpy as np
from matplotlib.colors import LogNorm
from sklearn import mixture
mpl.style.use('fivethirtyeight')
n_samples = 300

# generate random sample, two components
np.random.seed(0)

# generate spherical data centered on (20, 20)
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])

# generate zero centered stretched Gaussian data
C = np.array([[0., -0.7], [3.5, .7]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)

# concatenate the two datasets into the final training set
X_train = np.vstack([shifted_gaussian, stretched_gaussian])

# fit a Gaussian Mixture Model with two components

clf = mixture.GaussianMixture(max_iter=300, n_components=2, covariance_type='full')
clf.fit(X_train)

# display predicted scores by the model as a contour plot
x = np.linspace(-20., 30.)
y = np.linspace(-20., 40.)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T



Z = clf.score_samples(XX)
Z = Z.reshape(X.shape)
fig = plt.figure()
ax = fig.gca(projection='3d')
CS = ax.plot_surface(X, Y, Z, alpha=0.2)
ax.scatter(X_train[:, 0], X_train[:, 1], .8)

plt.title('GMM')
plt.axis('tight')
plt.show()