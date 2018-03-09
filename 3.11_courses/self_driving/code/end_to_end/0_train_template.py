import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import cv2
import pylab
import tensorflow as tf 
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda
from keras.layers import Conv2D, Dropout
from keras import losses 

np.random.seed(28)

# set the basic parameters
import params, utils

data_dir = params.data_dir
out_dir = params.out_dir
model_dir = params.model_dir
img_dir = os.path.abspath(params.data_dir + '/images/img/')
img_height = params.img_height
img_width = params.img_width
img_channels = params.img_channels
batch_size = params.batch_size

# (1) 加载数据
# load RGB data
import pickle

# (2) 数据分布
# 可视化Steering Angel的分布。
# 上图表明Steering Angle大致成正态分布，小幅转角出现的次数多，大幅转角出现的次数少。


# # 转角变化幅度
# # 可视化转角随时间的变化。


# (3) 数据打乱处理
# 利用train_test_split函数将样本数据进行打乱处理，增加数据的随机性。
# shuffle train data 

# (4) 构建模型


# (5) 基准模型构建
# 选用nvidia model作为基准模型。
# NVIDIA model

# (6) 模型训练
# Train the model with YUV data 

# (7) 在线推断
# Test the performance on test data

# (8) 可视化结果比对
# viualize model

