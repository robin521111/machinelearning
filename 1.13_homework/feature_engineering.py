from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import numpy as np 
iris = load_iris()

data = iris.data
print(data)
data_new = data[:,0]*data[:,1]
np.delete(data,[0,1],axis=0)

print(data)
np.insert(data,0,data_new)
