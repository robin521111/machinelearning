
# by cangye@hotmail.com
# TensorFlow入门实例

import os
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
mnist = input_data.read_data_sets("data/", one_hot=True)

#卷积函数
def conv2d_layer(input_tensor, size=1, feature=128, name='conv1d'):
    with tf.variable_scope(name):
        shape = input_tensor.get_shape().as_list()
        kernel = tf.get_variable('kernel', 
                                  (size, size, shape[-1], feature), 
                                  dtype=tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))#初始化值很重要，不好的初始化值比如文章中的初始化值会使得迭代收敛极为缓慢。
        b = tf.get_variable('b', [feature], dtype=tf.float32, initializer=tf.constant_initializer(0))
        out = tf.nn.conv2d(input_tensor, kernel, strides=[1, 2, 2, 1], padding='SAME') + b
    return tf.nn.relu(out)
#全链接函数
def full_layer(input_tensor, out_dim, name='full'):
    with tf.variable_scope(name):
        shape = input_tensor.get_shape().as_list()
        W = tf.get_variable('W', (shape[1], out_dim), dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('b', [out_dim], dtype=tf.float32, initializer=tf.constant_initializer(0))
        out = tf.matmul(input_tensor, W) + b
    return out


with tf.variable_scope("input"):
    x = tf.placeholder(tf.float32, [None, 784], name="input_x")
    label = tf.placeholder(tf.float32, [None, 10], name="input_label")
    x2d = tf.reshape(x, [-1,28,28,1])
#第一层卷积
net = conv2d_layer(x2d, size=4, feature=32, name='conv1')
#加入池化层，用于提取特征
#加入batchnorm层，减少过拟合，增加梯度迭代有效性
#net = tf.contrib.layers.batch_norm(net)
net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#定义卷积层
net = conv2d_layer(net, size=4, feature=32, name='conv2')
#net = tf.contrib.layers.batch_norm(net)
net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#flatten层，用于将三维的图形数据展开成一维数据，用于全链接层
net = tf.contrib.layers.flatten(net)
y=full_layer(net, 10, name='full')

with tf.variable_scope("loss"):
    #定义loss函数
    ce=tf.square(label-tf.nn.sigmoid(y))
    loss = tf.reduce_mean(ce)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
#用于训练参数的保存
saver = tf.train.Saver()
#载入保存的权值
#saver.restore(sess, tf.train.latest_checkpoint('model'))
for itr in range(501):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, label: batch_ys})
    if itr % 10 == 0:
        print("step:%6d  accuracy:"%itr, sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        label: mnist.test.labels}))
        saver.save(sess, os.path.join(os.getcwd(), 'model','handwriting'), global_step=itr)