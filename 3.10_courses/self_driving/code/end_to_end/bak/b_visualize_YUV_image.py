import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import cv2
import pylab

np.random.seed(28)

# set the basic parameters
import params, utils

data_dir = params.data_dir
out_dir = params.out_dir
model_dir = params.model_dir
img_dir = os.path.abspath(params.img_dir + '/img/')

img_height = params.img_height
img_width = params.img_width
img_channels = params.img_channels
batch_size = params.batch_size

# 加载数据
# load RGB data
import pickle

# load YUV data
pickle_file2 = 'data2.pickle'

with open ('data2.pickle', 'rb') as f:
    data = pickle.load(f)
    imgs_train_YUV = data['imgs_train_YUV']
    wheels_train_YUV = data['wheels_train_YUV'] 
    imgs_test_YUV = data['imgs_test_YUV']
    wheels_test_YUV = data['wheels_test_YUV']

print(imgs_train_YUV.shape, wheels_train_YUV.shape)
print(imgs_test_YUV.shape, wheels_test_YUV.shape)

# visualize the processed picture YUV
plt.imshow(imgs_train_YUV[0])
pylab.show()
