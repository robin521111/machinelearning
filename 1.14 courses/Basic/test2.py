import numpy as np
import tensorflow as tf

x = np.random.normal(0, 1, [1000, 1])
y = 2*x + 1 + np.random.normal(0, 0.1, [1000, 1])


# x, y
# y = wx+b

x_in = tf.placeholder(tf.float64, [5, 1])
y_in = tf.placeholder(tf.float64, [5, 1])
w = tf.Variable(np.random.random([1, 1]))
b = tf.Variable(np.random.random([1]))
y_out = tf.matmul(x_in, w) + b
loss = tf.reduce_mean(tf.square(y_in - y_out))
# ... 
step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for itr in range(500):
    idx = np.random.randint(0, 1000, 5)
    #print(np.shape(x[idx]))
    sess.run(step, feed_dict={x_in:x[idx], y_in:y[idx]})
    if itr%10 == 0:
        print(sess.run([loss, w.value(), b.value()], feed_dict={x_in:x[idx], y_in:y[idx]}))
