import tensorflow as tf
import numpy as np 

cst1 =tf.constant(np.ones([4,4]))
cst2 =tf.constant(np.ones([4,4]))
rest = tf.matmul(cst1,cst2)
sess = tf.Session()
var1=tf.Variable(np.ones([4,4]))

rest =tf.matmul(cst1,var1)

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(rest))
