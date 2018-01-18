from __future__ import print_function
import numpy as np 
from sklearn.preprocessing import Normalizer

data = [[-1,2],[-0.5,6],[0,10],[1,18]]
scaler = Normalizer()

print(scaler.fit(data))
print('test')
print(scaler.transform(data))
print('test')
print(scaler.transform([[2,2]]))
