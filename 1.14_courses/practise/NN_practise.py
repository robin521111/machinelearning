import tensorflow as tf 
import numpy as np 


with tf.variable_scope("fisrt-nn-layler") as scope:
    W = tf.get_variable('W1',[])