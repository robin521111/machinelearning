print(__doc__)
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

iris = datasets.load_iris()

#select two property 

X = iris.data 
y = iris.target 

pca = PCA(n_components=2)
X = pca.fit(X).transform(X)
gnb = GaussianNB()
gnb.fit(X,y)
import matplotlib.pyplot as plt 
import matplotlib as mpl 
import numpy as np 

mpl.style.use('fivethirtyeight')
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

pdt = gnb.predict_proba(np.c_[xx.ravel(), yy.ravel()])

Z = pdt[:, 0]
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.6)
Z = pdt[:, 1]
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.6)
#绘制散点图
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title("GaussianNaiveBayes")
plt.axis("equal")
plt.show()