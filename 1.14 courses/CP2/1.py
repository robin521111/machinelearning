import tensorflow as tf
import numpy as np
a1 = tf.constant(np.ones([4, 4]))
a2 = tf.constant(np.ones([4, 4]))
a1_dot_a2 = tf.matmul(a1, a2)

sess = tf.Session()
print(sess.run(a1_dot_a2))