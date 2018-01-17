import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

iris = datasets.load_iris()

X=iris.data
Y=iris.target

print(iris.target)
print(iris.data)


X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=.75)


print(X_train)
print('=========')
print(X_test)
print('=========')

print(Y_train)
print('=========')

print(Y_test)


classifier = KNeighborsClassifier(n_neighbors=5)

classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(Y_pred, Y_test)
accPer = '{:.1%}'.format(acc)


sepalLength = input("Enter a Sepal Length: ")
sepalWidth = input("Enter a Sepal Width: ")
petalLength = input("Enter a Petal Length: ")
petalWidth = input("Enter a Petal Width: ")

guess = iris.target_names[knn.predict([[sepalLength, sepalWidth, petalLength, petalWidth]])]

print("The probably species is %s. With a test accuracy of %s"%(guess,accPer))
