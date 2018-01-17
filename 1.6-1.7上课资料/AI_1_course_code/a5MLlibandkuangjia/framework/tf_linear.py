#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
 
# 定义输入数据
X_data = np.linspace(-1, 1, 100)
noise = np.random.normal(0, 0.5, 100)
y_data = 5 * X_data + noise
 
# Plot 输入数据
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(X_data, y_data)
plt.ion()
plt.show()
 
# 定义数据大小
n_samples = 100
 
# 转换成向量
X_data = np.reshape(X_data, (n_samples, 1))
y_data = np.reshape(y_data, (n_samples, 1))
 
# 定义占位符
X = tf.placeholder(tf.float32, shape=(None, 1))
y = tf.placeholder(tf.float32, shape=(None, 1))
 
# 定义学习的变量
W = tf.get_variable("weight", (1, 1),
                    initializer=tf.random_normal_initializer())
b = tf.get_variable("bais", (1,),
                    initializer=tf.constant_initializer(0.0))
y_pred = tf.matmul(X, W) + b
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_pred)))
 
# 梯度下降
# 定义优化函数
opt = tf.train.GradientDescentOptimizer(0.001)
operation = opt.minimize(loss)
 
with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.initialize_all_variables())
     
    lines = None
    for i in range(50):  
        _, loss_val = sess.run([operation, loss], 
                               feed_dict={X: X_data, y: y_data})
        
        if i % 5 == 0:
            if lines:
                ax.lines.remove(lines[0])
                 
            prediction_value = sess.run(y_pred, feed_dict={X: X_data})
            lines = ax.plot(X_data, prediction_value, 'r-', lw=5)
            plt.pause(1)
     
    plt.pause(50)