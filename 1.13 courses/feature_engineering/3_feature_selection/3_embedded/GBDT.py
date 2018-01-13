from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.datasets import load_iris
iris = load_iris()
#GBDT作为基模型的特征选择
#print(SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data, iris.target))
selector = SelectFromModel(GradientBoostingClassifier()).fit(iris.data, iris.target)
data = selector.transform(iris.data)
print(data)
print(selector.estimator_.feature_importances_)