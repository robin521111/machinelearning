import numpy as np 

import matplotlib.pyplot as plot 

import  tensorflow as tf 

x=np.random.normal(0,1,[10000,1])

y=2*x +1 +np.random.rand(1,0.1,[10000,1])

# plot.scatter(x,y)
# plot.show()


x_in = tf.placeholder(tf.float64,[5,1])
y_in =tf.placeholder(tf.float64,[5,1])

w=tf.Variable(np.random.rand([1, 1]))
b=tf.Variable(np.random.rand([1]))
y_out =tf.matmul(x_in,w) +b

loss =tf.reduce_mean(tf.square(y_in -y_out)) 
step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
sess = tf.Session()

sess.run(tf.global_variables_initializer())
for iter in range(300):
    idx=np.random.randint(0,1000,5)
    sess.run(step,feed_dict={x_in:[x[idx]],y_in:[y[idx]]})
    if itr%10 == 0:
        print(sess.run([loss,w.value(),b.value()],feed_dict={x_in:x[idx],y_in:y[idx]}))
