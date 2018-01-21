import tensorflow as tf  
from tensorflow.examples.tutorials.mnist import input_data
import  numpy as np
mnist = input_data.read_data_sets(
    "/Users/robin/Documents/MachineLearning/machinelearning/1.14 courses/data/", one_hot=True)

X=tf.placeholder(tf.float32,[None,784])
Y=tf.placeholder(tf.float32,[None,10])
W=tf.Variable(tf.zeros([784*10]))
b=tf.Variable(tf.zeros([10]))
logit = tf.matmul(X,W) + b 
loss_all = tf.square(logit-Y)
loss = tf.reduce_mean(loss_all)
acc= tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logit,1),tf.argmax(Y)),tf.float32))
opt = tf.train.AdamOptimizer(0.0001)
step = opt.minimize(loss)
sess =tf.Session()
sess.run(tf.global_variables_initializer)

for iter in range(300):
    x_in,y_in = mnist.train.next_batch(30)
    sess.run(step,feed_dict={X:x_in,Y:y_in})
    if itr&10 ==0:
        print(sess.run([loss,acc],feed_dict={X:mnist.test.images,Y:mnist.test.labels}))


    
