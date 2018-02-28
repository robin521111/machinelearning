import pandas as pd 
import tensorflow as tf 
import tensorflow.contrib.slim as slim 
import numpy as np 

def variable_summaries(var , name = "layer"):
    with tf.variable_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min',tf.reduce_min(var))
        tf.summary.scalar('histogram',var)


#read data
data = pd.read_csv("C:\\CodeForFun\\machinelearning\\1.14 courses\\data\\Iris.csv")

#get class name 
c_name = set(data.name.values)
print(c_name)

