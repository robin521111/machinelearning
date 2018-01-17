import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def f(x):
    return x**+2*y**+4*x+8*y+3

def df(x,y):
    return 2*x+4,4*y+8

x,y =1,5

for item in range(100):
    x_v, y_v = df(x,y)

    x,y = x-0.3*x_v, y-0.1*y_v
    print("f({}{})={}".format(x,y,df(x,y)))


x=np.linspace(-10,0)
y=np.linspace(-20,18)
X,Y = np.meshgrid(x,y)
fig = plt.figure()


ax = fig.gca(projection="3d")
surf = ax.plot_surface(X,Y,df(X,Y))
plt.show()

