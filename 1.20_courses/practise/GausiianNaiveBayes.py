print(__doc__)

from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_moons, make_circles,make_classification

#insert training data 
X, y = make_classification(n_features=2,n_redundant=2,n_informative=2,random_state=1,n_clusters_per_class=1)

#identify gausse classifier 
gnb = GaussianNB()

#training
gnb.fit(X,y)

#paint the chart
import matplotlib.pyplot as plt 
import matplotlib as mpl 
import numpy as np 

#adjust image style

mpl.style.use('fivethirtyeight')

#identify xy grid,

x_min, x_max = X[:,0].min() - .5, X[:, 0].max() + .5
y_min, y_max = Y[:,1].min() - .5, X[:, 1].max() + .5

xx, yy = np.meshgrid(np.arange(x_min,x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = gnb.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha= .8)

#paint scatter chart
plt.scatter(X[:,0],X[:,1],c=y)
plt.title("GaussianNaiveBayes")
plt.axis("equal")
plt.show()