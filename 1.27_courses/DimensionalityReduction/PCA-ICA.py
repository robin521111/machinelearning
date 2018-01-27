#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
观测三种降维方法所得的数据分布
ICA方法分布不相同。
PCA近似正态分布
字典集中于0的部分较多
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.decomposition import FastICA, PCA, DictionaryLearning

# #############################################################################
# Generate sample data
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

# Compute ICA
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix
# For comparison, compute PCA
pca = PCA(n_components=3)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components
print("PCA:", np.sum(H[:, 0]*H[:, 1]))
# Compute DL
dl = DictionaryLearning(n_components=3)
D = dl.fit_transform(X)  # Reconstruct signals based on orthogonal components

# #############################################################################
# Plot results
print("Ploting...")
plt.figure()


models = [X, S, S_, H, D]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals',
         'PCA recovered signals',
         'DL recovered signals']
colors = ['red', 'steelblue', 'orange', 'blue']
plt.figure(1)
for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(5, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)


plt.figure(2)
for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(5, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.hist(sig, color=color, alpha = 0.3)

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)


plt.show()