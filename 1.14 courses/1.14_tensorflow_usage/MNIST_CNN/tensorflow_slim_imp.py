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
    x2d = tf.reshape(x, [-1,28,28,1])
#定义卷积层
net = slim.conv2d(x2d, 4, 16, stride=2, activation_fn=tf.nn.relu, scope='conv1')
#加入池化层，用于提取特征
net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
net = slim.conv2d(net, 4, 32, stride=2, activation_fn=tf.nn.relu, scope='conv2')
net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#flatten层，用于将三维的图形数据展开成一维数据，用于全链接层
net = slim.flatten(net)
y = slim.fully_connected(net, 10, activation_fn=line, 
                              scope='full', reuse=False)

with tf.variable_scope("loss"):
    #定义loss函数
    ce=tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=y)
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