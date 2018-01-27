from sklearn.svm import SVC
import numpy as np 
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import PCA

train = np.load()
test = np.load()
pca = PCA(n_components=20)

X_r = pca.fit(train["images"]).transform(train["images"])
vect_r = np.array([[itr] for itr in range(10)])
X_train = X_r[:6000]
y_train = np.dot(train["labels"][:6000],vect_r).ravel().astype("int")
X_test = X_r[6000:12000]
Y_test = np.dot(train["labels"][6000:12000],vect_r).ravel().astype("int")


print(np.shape(train["images"]), np.shape(X_test))

estimator = KNeighborsClassifier(5) 
estimator.fit(X_train,y_train)
result = estimator.predict_proba(X_test)
print(np.sum(result== y_test)/6000.)

