print(__doc__)
# Code source: Jaques Grobler
# License: BSD 3 clause
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
# Load the diabetes dataset
# scikit learn规定 w0是 intercept_. w1--wp是coef_.
diabetes = datasets.load_diabetes()

print("1")
print(diabetes.data[0:5].shape)
# Use only one feature
print("2")
print(diabetes.data[0:5, np.newaxis].shape)
diabetes_X = diabetes.data[:, np.newaxis, np.newaxis, 2]
print("3")
print(diabetes_X[0:5].shape)

diabetes_X = diabetes.data[:, 2:4]
print("4")
print(diabetes_X[0:5])