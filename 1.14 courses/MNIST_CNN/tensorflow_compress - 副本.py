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
inputs = tf.placeholder(tf.float32, [None, None, None, 1], name="input_x")
#定义卷积层
net = slim.conv2d(inputs, 32, 7, activation_fn=tf.nn.relu, padding="VALID")
net = tf.contrib.layers.batch_norm(net)
net = slim.conv2d(net, 32, 7, activation_fn=tf.nn.relu, padding="VALID")
net = tf.contrib.layers.batch_norm(net)
net = slim.conv2d(net, 32, 7, activation_fn=tf.nn.relu, padding="VALID")
net = tf.contrib.layers.batch_norm(net)
net = slim.conv2d(net, 32, 7, activation_fn=tf.nn.relu, padding="VALID")
net = tf.contrib.layers.batch_norm(net)
net = slim.conv2d(net, 32, 4, activation_fn=tf.nn.relu, padding="VALID")
net = tf.contrib.layers.batch_norm(net)
net3 = slim.conv2d(net, 3, 1, activation_fn=tf.nn.sigmoid, padding="VALID")
cprs = tf.squeeze(net3)
net = tf.contrib.layers.batch_norm(net3)
net = slim.conv2d(net, 10, 1)
net = tf.contrib.layers.batch_norm(net)
softmax = tf.nn.softmax(net, dim=-1)
print("SHAPE", net.get_shape().as_list())
y = tf.squeeze(net)
#flatten层，用于将三维的图形数据展开成一维数据，用于全链接层


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
saver.restore(sess, tf.train.latest_checkpoint('model/full'))
"""
for itr in range(500):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, label: batch_ys})
    if itr % 10 == 0:
        print("step:%6d  accuracy:"%itr, sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        label: mnist.test.labels}))
        saver.save(sess, os.path.join(os.getcwd(), 'model','full','handwriting'), global_step=itr)
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation

image = np.zeros([1, 200, 200, 1])
image[0, 90:90+28, 90:90+28, 0] = np.reshape(mnist.test.images[0], [28, 28])
for itr in range(100):
    vvv=np.reshape(mnist.test.images[itr], [1, 28, 28, 1])
    vect0 = sess.run(y, feed_dict={inputs:vvv})
    #print(vect0, mnist.test.labels[itr])
    vect0 = vect0/np.sqrt(np.sum(np.square(vect0)))
    print(vect0, mnist.test.labels[itr])

plt.figure(1)
plt.matshow(image[0, :, :, 0])
plt.figure(2)
out = sess.run(cprs, feed_dict={inputs: image})
ix, iy, _ = np.shape(out)
mats = np.zeros([ix, iy])
for idx, emu in enumerate(out):
    for idy, vct in enumerate(emu):
        vect1 = vct/np.sqrt(np.sum(np.square(vct)))
        mats[idx, idy] = np.sum(np.square(vect1-vect0))
print()
plt.matshow(mats)
plt.colorbar()

"""
fig = plt.figure(3)
ax = fig.gca(projection='3d')
xx = yy = np.linspace(0, 1, len(out[0, 0, :, 0]))
X, Y = np.meshgrid(xx, yy)
print(np.shape(out))
ax.plot_surface(X, Y, out[0, :, :, 7], alpha=1)
#plt.matshow(out[0, :, :, 0])
"""
plt.show()