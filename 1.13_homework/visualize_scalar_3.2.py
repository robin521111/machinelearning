import matplotlib.pyplot as plt 
import numpy as np 

X = np.linspace(-2, 2, 20)
Y = 2 * X +1
T=Y 
plt.scatter(X,Y,c=T)
plt.show()