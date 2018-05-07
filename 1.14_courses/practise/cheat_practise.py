import pandas as pd 
import tensorflow as tf 
import numpy as np 

data = pd.read_csv("C:\CodeForFun\\machinelearning\\1.14 courses\\data\\creditcard.csv")

class1 = data[data.Class == 0]
print(len(class1))
print(np.shape(class1.values))


#identify NN

x = tf.placeholder(tf.float32,[None,32],name="input_x")
label = tf.placeholder(tf.float32,[None,2], name="input_y")

#identify weight

W1 = tf.get_variable('W1', [3,32], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))

b1 = tf.get_variable('b1',[32],dtype=tf.float32,initializer=tf.constant_initializer(0))

h1 = tf.nn.sigmoid(tf.matmul(x,W1)+b1)


#identify secod lyler of NN

W2 = tf.get_variable('W1',[32,32],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))


b2 = tf.get_variable('b2',[])


