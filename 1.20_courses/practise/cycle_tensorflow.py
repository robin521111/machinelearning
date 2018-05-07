import tensorflow as tf 
import tensorflow.contrib.slim as slim 
import pandas as pd 
import  numpy as np 
from sklearn import datasets
from sklearn.datasets import make_circles, make_moons, make_classification

X, y  = make_circles(noise=0.1, factor=0.5, random_state=1, n_samples=10000)

y_r = np.zeros([len(X), 2])
print(y_r)

#create CNN

x_tf = tf.placeholder(tf.float32, [None,2])
label_tf = tf.placeholder(tf.float32, [None,2])

x2x = tf.concat([x_tf, x_tf[:,:1]*x_tf[:,1:2], x_tf**2],axis=1)
y_tf = slim.fully_connected(x2x, 2, scope='full', activation_fn = tf.nn.sigmoid, reuse= False)

ce = tf.nn.softmax_cross_entropy_with_logits(labels=label_tf, logits=y_tf))
loss = tf.reduce_mean(ce)
train_step = tf.train.AdamOptimizer(0.005).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for itr in range(6000):
    sess.run(train_step, feed_dict={x_tf: X, label_tf: y_r})

#imvovle matplotlib for chart 
import matplotlib.pyplot as plt 
import matplotlib as mpl 
import numpy as np 

mpl.style.use('fivethirtyeight')

#identify xy grid, use for chart making

x_min, x_max = X[:, 0].min() - .5, X[:,0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:,1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
np.arange(y_min, y_max, 0.1))

pdt = sess.run(y_tf, feed_dict={x_tf: np.c_[xx.ravel(), yy.ravel()]})