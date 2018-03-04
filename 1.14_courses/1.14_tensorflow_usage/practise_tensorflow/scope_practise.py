import tensorflow as tf 
with tf.variable_scope("first-nn-layler"):
    W = tf.Variable(tf.zeros([784,10]),name="W")
    b = tf.Variable(tf.zeros([10]),name="b")
    W1 = tf.Variable(tf.zeros([784,10]),name="W")
print(W.name)
print(W1.name)
