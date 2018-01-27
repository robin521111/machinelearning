print(__doc__)

from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons,make_circles,make_classification

X,y = make_classification(n_features=2, n_redundant=0,n_informative=2,random_state=1,n_clusters_per_class=1,class_sep=.2)

rdf = KNeighborsClassifier(n_neighbors=1)
rdf = BaggingClassifier(base_estimator=rdf, n_estimators=100, max_samples=50)

rdf.fit(X,y)

import matplotlib.pyplot as  plt 
import matplotlib as mpl 
import numpy as np 

mpl.style.use('fivethirtyeight')
x_min, x_max = X[:,0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:,1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min,y_max,0.1))

Z = rdf.predict_proba(np.c_[xx.ravel(),yy.ravel()])[:,1]
Z = Z.reshape(xx.shape)
plt.contourf(xx,yy,Z,alpha=.7)

plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title("Bagging")
plt.axis("equal")
plt.show()
