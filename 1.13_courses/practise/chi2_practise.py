from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.datasets import load_iris

iris = load_iris()

selector = SelectKBest(chi2,k=2).fit(iris.data, iris.target)
data = selector.transform(iris.data)
print(data)
