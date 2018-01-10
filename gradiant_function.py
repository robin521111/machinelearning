import numpy as np 
import matplotlib.pyplot as plt

def f(x):
    return x**2+4*x+4

def df(x):
    return 2*x+4

x_old = 1.9
x=[]


for iter in range(20):
    x_new = x_old-0.6* df(x_old)
    x_old=x_new
    print("f({})={}".format(x_new,f(x_new)))
    
x=np.linspace(-10,9)
plt.plot(x,f(x))
plt.show()