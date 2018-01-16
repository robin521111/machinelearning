from sklearn import preprocessing
from sklearn.datasets import load_iris
import numpy as np 
from sklearn.preprocessing import StandardScaler

iris = load_iris()

data = iris.data 
target = iris.target 

# print('before processing')
# print(data)
# standard_data = StandardScaler().fit_transform(iris.data)

# print('after processing')
# print(standard_data)

X_train = np.array([[1.,-1.,2.],[2.,0.,0],[0.,1.,-1.]])
X_Scaled = preprocessing.scale(X_train)
print(X_Scaled)
X_Scaled.mean(axis=0)
print(X_Scaled)