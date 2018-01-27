import matplotlib.pyplot as plt
import numpy as np 

x = np.linspace(-16,16,50)
y=x*x*x + 10
plt.figure(figsize=(8,8))
plt.plot(x,y)
plt.show()

