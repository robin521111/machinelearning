import tensorflow as tf 
import pandas as pd 
import numpy as np 

data = pd.read_csv("D:\\CodeForFun\\machinelearning\\1.14 courses\\data\\creditcard.csv")

class1 = data[data.Class == 0]
class2 = data[data.Class == 1]
print(len(class1))
print(len(class2))

data1 = class1.values
data2 = class2.values 

# print(data1.describe)

x = tf.placeholder(tf.float32,[None,28],name="input_x")
label = tf.placeholder(tf.float32,[None,2],name="input_y")


W1 = tf.get_variable("W1", [28,28],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
b1 = tf.get_variable("b1",[28],dtype=tf.float32,initializer=tf.constant_initializer(0))
h1 = tf.nn.sigmoid(tf.matmul(x,W1)+b1)


W2 = tf.get_variable("W2",[28,2],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
b2 = tf.get_variable("b2",[10],dtype=tf.float32,initializer=tf.constant_initializer(0))
h1 = tf.nn.sigmoid(tf.matmul(x,W2)+b2)

w3 = tf.get_variable("W3",[28,])