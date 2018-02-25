import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


y_pred= [0.25,0.25,1,0.25]


y_test= [0.5,1,0.5,0.5]


print("mean square error: %.2f" % mean_squared_error(y_test,y_pred))


                     