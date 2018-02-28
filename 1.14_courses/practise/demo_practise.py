import pandas as pd 
import tensorflow as tf 
import numpy as np 

data = pd.read_csv("D:\\CodeForFun\\machinelearning\\1.14 courses\\data\\creditcard.csv")

class1 = data[data.Class == 0]
class2 = data[data.Class ==1]

print(len(class1))
print(len(class2))
print(np.shape(class1.values))


data1 = class1.values
data2 = class2.values

x = tf.placeholder(tf.float32,[None,28],name="input_x")
label = tf.placeholder(tf.float32,[None,2])