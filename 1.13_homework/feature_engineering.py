from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import numpy as np 
iris = load_iris()

data_new = iris.data[:, 0] * iris.data[:, 1]

new_feature = iris.data[:, :2]
# print(new_feature)
np.append(new_feature,data_new)
print(new_feature)


# print(data_row)
# np.delete(iris.data,2,axis=None)
# print(iris.data)
# np.insert(iris.data,0,data_new)
# print(iris.data)
