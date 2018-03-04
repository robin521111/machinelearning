import numpy as np 


a = np.reshape([1,2,3,4],[1,-1])
b = np.reshape([1,2,3,4],[-1,1])
print('this is b',b)
print(a)