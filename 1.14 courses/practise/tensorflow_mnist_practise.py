from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf 

# mnist = input_data.read_data_sets("1.14 courses\data",one_hot=True)
# #establish the CNN
# x = tf.placeholder(tf.float32,[None,784])
# label = tf.placeholder(tf.float32,[None,10])
# #create weight and asset
# # W = tf.placeholder()
# correct_prediciton = tf.equal(tf.argmax(x,1),tf.argmax(label,1))
# print(correct_prediciton)

#identify full connected layer

def full_layer(input_tensor, out_dim, name = 'full'):
    with tf.variable_scope(name):
        shape = input_tensor.get_shape().as_list()
        W = tf.get_variable('W', (shape[1], out_dim), dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('b', [out_dim], dtype=tf.float32, initializer=tf.constant_initializer(0))
        out = tf.matmul(input_tensor, W) + b
    return tf.nn.sigmoid(out) 

def model(net, out_dim):
    net = full_layer(net, out_dim, "full_layer1")
    return net


#identify input
with tf.variable_scope("input"):
    x = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 10])

#identify model
y = model(x, 10)
#identify loss function
loss = tf.reduce_mean(tf.square(y-label))
#gradient descent optimization
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#identify varification function 
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(label,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#identify session 
sess = tf.Session()
#initialize all variant
sess.run(tf.global_variables_initializer())
train_write = tf.summary.FileWriter("practise-mnist-logdir",sess.graph)

mnist = input_data.read_data_sets("data/", one_hot=True)



    