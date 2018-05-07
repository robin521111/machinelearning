import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import datasets

iris  = datasets.load_iris()

X = iris.data
Y = iris.target

# print(iris.target)
# print(iris.data)

X_train, Y_train, X_test, Y_test = train_test_split(X,Y, test_size=.75)
# print(X_train)
# print(X_test)


classifier = KNeighborsClassifier(n_neighbors=5)

classifier.fit(X_train, Y_train)

Y_predict = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(Y_predict, Y_test)

accPer = '{:.1%}'.format(acc)

print(classifier.score(X_test, Y_test))

