# by cangye@hotmail.com

import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

data = pd.read_csv("IRIS/iris.data.csv")
c_name = set(data.name.values)
print(c_name)
iris_label = np.zeros([len(data.name.values),len(c_name)])
iris_data = data.values[:, :-1]
iris_data = iris_data-np.mean(iris_data, axis=0)
iris_data = iris_data/np.max(iris_data, axis=0)
train_data=[]
train_data_label=[]
test_data=[]
test_data_label=[]
for idx, itr_name in enumerate(c_name):
    datas_t = iris_data[data.name.values==itr_name, :]
    labels_t = np.zeros([len(datas_t), len(c_name)])
    labels_t[:, idx] = 1
    train_data.append(datas_t[:10])
    train_data_label.append(labels_t[:10])
    test_data.append(datas_t[10:])
    test_data_label.append(labels_t[10:])
train_data = np.concatenate(train_data)
train_data_label = np.concatenate(train_data_label)
test_data = np.concatenate(test_data)
test_data_label = np.concatenate(test_data_label)
x = tf.placeholder(tf.float32, [None, 4], name="input_x")
label = tf.placeholder(tf.float32, [None, 3], name="input_y")
# 对于sigmoid激活函数而言，效果可能并不理想
net = slim.fully_connected(x, 4, activation_fn=tf.nn.relu, 
                              scope='full1', reuse=False)
net = slim.fully_connected(net, 12, activation_fn=tf.nn.relu, 
                              scope='full4', reuse=False)
net = slim.fully_connected(net, 12, activation_fn=tf.nn.relu, 
                              scope='full5', reuse=False)
y = slim.fully_connected(net, 3, activation_fn=tf.nn.sigmoid, 
                              scope='full6', reuse=False)
loss = tf.reduce_mean(tf.square(y-label))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
optimizer = tf.train.GradientDescentOptimizer(0.5)
var_list = [var for var in tf.trainable_variables()]
gradient = optimizer.compute_gradients(loss, var_list=var_list)
for itr_g in gradient:
    variable_summaries(itr_g)
train_step = optimizer.apply_gradients(gradient)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
train_writer = tf.summary.FileWriter("logdir", sess.graph)
merged = tf.summary.merge_all()
for itr in range(600):
    sess.run(train_step, feed_dict={x: train_data, label: train_data_label})
    if itr % 30 == 0:
        acc1 = sess.run(accuracy, feed_dict={x: train_data,
                                        label: train_data_label})
        acc2 = sess.run(accuracy, feed_dict={x: test_data,
                                        label: test_data_label})
        print("step:{:6d}  train:{:.3f} test:{:.3f}".format(itr, acc1, acc2))
        summary = sess.run(merged, 
                           feed_dict={x: train_data,
                                        label: train_data_label})
        train_writer.add_summary(summary, itr)