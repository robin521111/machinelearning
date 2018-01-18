from sklearn.preprocessing import PolynomialFeatures
import numpy as np 

X =  np.arange(12).reshape(6,2)
print(X)
poly = PolynomialFeatures(2)
print(poly.fit_transform(X))