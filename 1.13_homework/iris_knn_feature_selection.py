from sklearn import feature_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
iris = load_iris()

X = iris.data 
y = iris.target 
# print(X)
selector = SelectKBest(chi2, k=2).fit(X, y)

data = selector.transform(X)


X_train , X_test ,Y_train , Y_test = train_test_split(X , y)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,Y_train)
Y_pred = knn.predict(X_test)
acc = accuracy_score(Y_pred,Y_test)
accPer = '{:.1%}'.format(acc)
print(knn.score(X_test,Y_test))
