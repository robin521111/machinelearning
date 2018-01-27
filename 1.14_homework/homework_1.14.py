import numpy as np 
import tensorflow as tf 


# x=tf.Variable(tf.random_uniform([1],-100.0,100.0,dtype=tf.double))
# y = tf.square(x)+2*y+tf.square(y)+4*x +8*y +3

# def f(x,y):
#     return x** + 2*y+y**+4*x + 8*y +3

# def df(x,y):
#     return 2*x +4, 2+2*y+8

# x,y = 3,6

W=tf.Variable([.3],dtype=tf.float32)
b = tf.Variable([-.3],dtype=tf.float32)




x_in = tf.placeholder(tf.float32)
y_in = tf.placeholder(tf.float32)

y_out = W*x_in +b 
loss = tf.reduce_mean(tf.square(y_in-y_out))

step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

for item in range(500):
    sess.run(step,feed_dict={x_in:x_train,y_in:y_train})

    curr_W,curr_b,curr_loss = sess.run([W,b,loss],{x_in:x_train,y_in:y_train})
    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))