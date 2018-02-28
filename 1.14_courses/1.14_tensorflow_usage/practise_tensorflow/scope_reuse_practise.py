import tensorflow as tf 
with tf.variable_scope("first-nn-layer") as scope:
    W=tf.get_variable("w",[784,10])
    b=tf.get_variable("b",[10])
    scope.reuse_variables()
    W1 = tf.get_variable("W",shape=[784,10])
print(W.name)
print(W1.name)
