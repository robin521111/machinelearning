import numpy as numpy
import matplotlib.pyplot as pyplot

def f_2d(x,y):
    return x**2+3*x+y**2+8*y+1

def df_2d(x,y):
    return 2*x+3, 2*y+8
x,y=3,6
for iter in range(200):
    x_v, y_v = df_2d(x,y)

    x,y = x-0.1*x_v, y-0.1*y_v
    print("f({}{})={}".format(x,y,f_2d(x,y)))



from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


x=np.linspace(-10,0)
y=np.linspace(-20,18)
X,Y=np.meshgrid(x,y)
fig=plt.figure()

ax=fig.gca(projection="3d")
surf=ax.plot_surface(X,Y,f_2d(X,Y))
plt.show()