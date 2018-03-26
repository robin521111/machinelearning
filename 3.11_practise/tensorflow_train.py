import tensorflow as tf 
import numpy as np 


x_data = np.linspace(-1, 1 , 300)[:, np.newaxis]

print(x_data)
noise = np.random.normal(0, 0.05, x_data.shape) 
y_data = np.square(x_data) - 0.5 + noise 
