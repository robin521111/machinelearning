import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
with tf.variable_scope("layer"):
    W = tf.Variable(tf.zeros([784, 10]), name="W")
    b = tf.Variable(tf.zeros([10]))
print(W.name)
logit = tf.matmul(X, W) + b
loss_all = tf.square(logit-Y)
loss = tf.reduce_mean(loss_all)

acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logit, 1),         tf.argmax(Y, 1)), 
        tf.float32))

opt = tf.train.AdamOptimizer(0.001)
step = opt.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
summ = tf.summary.FileWriter("first_logdir", sess.graph)
for itr in range(300):
    x_in, y_in = mnist.train.next_batch(30)
    sess.run(step, feed_dict={X: x_in, Y:y_in})
    if itr&10 == 0:
        print(itr, sess.run([loss, acc], feed_dict={X:mnist.test.images, Y:mnist.test.labels}))
        saver.save(sess, "first_model/one", global_step=itr)

saver.restore(sess, "first_model/one-289")
import matplotlib.pyplot as plt
for itr in range(10):
    plt.matshow(np.reshape(mnist.test.images[itr], [28, 28]))
    #plt.show()
out = sess.run(tf.argmax(logit, 1), feed_dict={X:mnist.test.images[:10]})
print(out)
    