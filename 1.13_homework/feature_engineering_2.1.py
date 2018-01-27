from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import numpy as np 
iris = load_iris()

data = iris.data
# print(data)
data_new = data[:,0]*data[:,1]
print(type(data_new))
v_data_new = data_new.reshape(150,1)

data_newfeature=iris.data[:,:2]

np.insert(data_newfeature,2,values=v_data_new,axis=1)

# np.column_stack((data_newfeature,v_data_new))
# data_newfeature = data_newfeature

# arr = np.zeros((150,),dtype=[('var1','var2','var3'),('var2','var2','var3')])
# arr['var1']= data_new[:,:1]
# arr['var2']= data_new[:,:2]

# arr['var3'] = v_data_new
# print(arr)


