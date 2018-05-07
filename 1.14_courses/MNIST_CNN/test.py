# by cangye@hotmail.com
# TensorFlow入门实例

import os
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
mnist = input_data.read_data_sets("data/", one_hot=True)

def line(net):
    return net
with tf.variable_scope("input"):
    x = tf.placeholder(tf.float32, [None, 784], name="input_x")
    label = tf.placeholder(tf.float32, [None, 10], name="input_label")
    x2d = tf.reshape(x, [-1,28,28,1])#[-, 56, 56, 256]
#定义卷积层
net = slim.conv2d(x2d, 10, 28, padding="VALID")
print(net.get_shape().as_list())