print(__doc__)


import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#load the diabetes datasets

diabetes = datasets.load_diabetes()

#Use only one feature 
diabetes_X = diabetes.data[:, np.newaxis, 2]

#splite the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

#splite the data into training/testing stes
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]


regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train,diabetes_y_train)


diabetes_y_pred = regr.predict(diabetes_X_test)

print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test,diabetes_y_pred))


plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
plt.plot(diabetes_X_test,diabetes_y_pred, color='blue', linewidth=4)
plt.xticks(())
plt.yticks(())
plt.show()