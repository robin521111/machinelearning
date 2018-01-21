
# DecisionTreeClassifier is a class capable of performing multi-class classification on a dataset.
# scikit-learn决策树算法类库内部实现是使用了调优过的CART树算法，
# 既可以做分类，又可以做回归。分类决策树的类对应的是DecisionTreeClassifier，
# 而回归决策树的类对应的是DecisionTreeRegressor。两者的参数定义几乎完全相同，但是意义不全相同。

# As with other classifiers, DecisionTreeClassifier takes as input two arrays: an array X, 
# of size [n_samples, n_features] holding the training samples,
# and an array Y of integer values, size [n_samples], holding the class labels for the training samples:
from sklearn import tree
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
# After being fitted, the model can then be used to predict the class of samples:
print(clf.predict([[2., 2.]]))
