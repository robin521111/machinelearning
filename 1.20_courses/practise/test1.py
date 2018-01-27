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

