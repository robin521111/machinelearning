import numpy as np
import matplotlib.pyplot as plt

N = 2000
y = 0
xs = []
ys = []

for i in np.arange(N):
    x = np.random.normal(0.8*y, 0.6)
    y = np.random.normal(0.8*x, 0.6)
    xs.append(x)
    ys.append(y)

xs2, ys2 = np.random.multivariate_normal( [0, 0], [[1,0.8],[0.8,1]], N ).T

plt.subplot(211)    
plt.scatter(xs, ys)
plt.subplot(212)
plt.scatter(xs2, ys2)
plt.show()


    