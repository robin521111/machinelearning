import tensorflow as tf 
import numpy as np 

a1 = tf.constant(np.ones([4,4])*2)
a2 = tf.constant(np.ones([4,4]))

b1 = tf.Variable(a1)
b2 = tf.Variable(np.ones([4,4]))

a1_elementwise_a2= a1*a2

a1_dot_a2=tf.matmul(a1,a2)

b1_elementwise_b2 = b1*b2
b1_dot_b2 = tf.matmul(b1,b2)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print(sess.run(a1_elementwise_a2))
print(sess.run(a1_dot_a2))
print(sess.run(b1_elementwise_b2))
print(sess.run(b1_dot_b2))
