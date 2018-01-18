from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
iris = load_iris()
selector = RFE(estimator=LogisticRegression(),n_features_to_select=2).fit(iris.data, iris.target)
data = selector.transform(iris.data)
print(data)
print(selector.ranking_)