import tensorflow as tf
import numpy as np

cst1 = tf.constant(np.ones([4, 4]))
cst2 = tf.constant(np.ones([4, 4]))
var1 = tf.Variable(np.ones([4, 4]))
plc1 = tf.placeholder(tf.float32, [4, 4])
plc1_trans = tf.cast(plc1, tf.float64)
rest = tf.matmul(cst1, var1)
rest2 = tf.matmul(plc1_trans, var1)
add1 = cst1+var1
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
print(sess.run(add1))
#print(sess.run(add1, feed_dict={plc1:np.ones([4, 4])}))

