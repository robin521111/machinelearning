from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf 
#identify full connected layler

def full_layler(input_tensor, out_dim, name="full"):
    with tf.variable_scope(name):
        shape= input_tensor.get_shape().as_list()
        W = tf.get_variable('W', (shape[1], out_dim), dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('b', [out_dim], dtype=tf.float32, initializer=tf.constant_initializer(0))
        out = tf.matmul(input_tensor,W)+b
    return tf.nn.sigmoid(out)


def model(net,out_dim):
    net = full_layler(net,out_dim,"full_layler1")
    return net

with tf.variable_scope("inputs"):
    x = tf.placeholder(tf.float32,[None,784])
    label = tf.placeholder(tf.float32,[None,10])

#involve mole
y = model(x,10)


loss = tf.reduce_mean(tf.square(y-label))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(label,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
sess = tf.Session()

sess.run(tf.global_variables_initializer())
train_write = tf.summary.FileWriter("mnist-log-robin",sess.graph)
mnist = input_data.read_data_sets("D:\CodeForFun\machinelearning\Mnist_data",one_hot=True)
for iter in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs, label:batch_ys})
    if  iter % 10 ==0:
        print("step:%6d accuracy:"%iter, sess.run(accuracy,feed_dict={x:mnist.test.images,label:mnist.test.labels}))


#below part is using for chart


    