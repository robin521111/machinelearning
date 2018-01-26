print(__doc__)


import tensorflow as tf 
import tensorflow.contrib.slim as slim 
import pandas as pd 
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.datasets import make_moons, make_circles,make_classification


#get data
X, y = make_circles(noise=0.2, factor=.5,random_state=1)

y_r = np.zeros([len(X),2])

# print(y_r)
print(y) 
for idx, itr in enumerate(y):
    y_r[idx,itr] = 1 

#create NN network
# print(y_r)
x_tf = tf.placeholder(tf.float32,[None,2])
label_tf = tf.placeholder(tf.float32,[None,2])

x2x = tf.concat([x_tf,x_tf[:,:1]*x_tf[:,1:2],x_tf**2],axis=1)

y_tf = slim.fully_connected(x2x,2,scope="full1", activation_fn=tf.nn.sigmoid,reuse=False)

ce = tf.nn.softmax_cross_entropy_with_logits(labels=label_tf,logits=y_tf)
loss = tf.reduce_mean(ce)

train_step = tf.train.AdamOptimizer(0.005).minimize(loss)

sess= tf.Session()
sess.run(tf.global_variables_initializer())

for itr in range(6000):
    sess.run(train_step,feed_dict={x_tf:X,label_tf:y_r})


import matplotlib.pyplot as plt 
import matplotlib as mpl 
import numpy as np 

mpl.style.use('fivethirtyeight')

x_min,x_max = X[:,0].min() - .5, X[:,0].max() + .5
y_min, y_max = X[:,1].min() - .5, X[:,1].max() + .5

xx, yy = np.meshgrid(np.arange(x_min,x_max,0.1), np.arange(y_min,y_max))

#prediect possibility

pdt = sess.run(y_tf, feed_dict={x_tf: np.c_[xx.ravel(), yy.ravel()]})
Z = pdt[:,1]
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.6)

plt.scatter(X[:,0], X[:, 1], c=y, edgecolors='k')
plt.title("NN")
plt.axis("equal")
plt.show()
