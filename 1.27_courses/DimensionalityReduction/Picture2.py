#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
三种方式进行降维后对图片进行恢复
最后一张图片是所恢复的人脸，其他为分量
"""
print(__doc__)


import logging
from time import time

from numpy.random import RandomState
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn import decomposition
import numpy as np

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
n_row, n_col = 3, 3
n_components = 30
image_shape = (64, 64)

# #############################################################################
# Load faces data
dataset = fetch_olivetti_faces(data_home="data", shuffle=True)
faces = dataset.data

n_samples, n_features = faces.shape

# global centering
faces_centered = faces - faces.mean(axis=0)

# local centering
faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

print("Dataset consists of %d faces" % n_samples)


def plot_gallery(title, images, data, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    my_face = np.dot(data, images)
    print(np.shape(data), np.shape(images), np.shape(my_face))
    for i, comp in enumerate(images[:9]):
        plt.subplot(n_row, n_col, i + 1)
        if i != 8:
            vmax = max(comp.max(), -comp.min())
            plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                    interpolation='nearest',
                    vmin=-vmax, vmax=vmax)
        else:
            comp = my_face[66]
            vmax = max(comp.max(), -comp.min())
            plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                    interpolation='nearest',
                    vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

# #############################################################################
# List of the different estimators, whether to center and transpose the
# problem, and whether the transformer uses the clustering API.
estimators = [
    ('PCA',
     decomposition.PCA(n_components=n_components),
     True),

    ('FastICA',
     decomposition.FastICA(n_components=n_components),
     True),

    ('MiniBatchDictionaryLearning',
        decomposition.MiniBatchDictionaryLearning(n_components=n_components),
     True)
]


for name, estimator, center in estimators:
    print("Extracting the top %d %s..." % (n_components, name))
    t0 = time()
    data = faces_centered
    trans_data = estimator.fit_transform(data)
    train_time = (time() - t0)
    print("done in %0.3fs" % train_time)
    components_ = estimator.components_
    plot_gallery('%s - Train time %.1fs' % (name, train_time),
                 components_, trans_data)

plt.show()