import tensorflow as tf 
import numpy as np 

cst1 = tf.constant(np.ones([4,4]))
cst2 = tf.constant(np.ones([4,4]))
var1 = tf.Variable(np.ones([4,4]))

plc1 = tf.placeholder(tf.float32,[4,4])
plc2_trans = tf.case(plc1,tf.float64)

rest = tf.matmul(cst1, var1)

add = cst1 + var1

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(sess.run(add))