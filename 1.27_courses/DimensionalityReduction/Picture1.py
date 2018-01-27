#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
对原始图片进行可视化
"""
print(__doc__)


from time import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn import decomposition as dcp

plot_grid = (3, 3)
image_shape = (64, 64)

dataset = fetch_olivetti_faces(data_home="data", shuffle=True)
faces = dataset.data

n_samples, n_features = faces.shape


plt.figure(1)
for itr in range(9):
    plt.subplot(3, 3, itr+1)
    plt.imshow(np.reshape(faces[itr+66], image_shape))

plt.show()