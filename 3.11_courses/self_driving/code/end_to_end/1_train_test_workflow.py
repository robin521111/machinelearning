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
img_dir = os.path.abspath(params.data_dir + '/images/img/')
img_height = params.img_height
img_width = params.img_width
img_channels = params.img_channels
batch_size = params.batch_size

# (1) 加载数据
# load RGB data
import pickle

import a_loading_training_RGB

imgs_train_RGB, wheels_train_RGB, imgs_test_RGB, wheels_test_RGB = a_loading_training_RGB.loadTrainingData()

import a_explore_RGB_data

a_explore_RGB_data.exploreData(imgs_train_RGB, wheels_train_RGB, imgs_test_RGB, wheels_test_RGB)
# (2) 数据分布
# 可视化Steering Angel的分布。
# 上图表明Steering Angle大致成正态分布，小幅转角出现的次数多，大幅转角出现的次数少。

import b_visualize_hist

b_visualize_hist.steering_distribution()
# # 转角变化幅度
# # 可视化转角随时间的变化。

import b_visualize_angel

b_visualize_angel.angel_visualize()
# (3) 数据打乱处理
# 利用train_test_split函数将样本数据进行打乱处理，增加数据的随机性。
# shuffle train data 
import a_shuffle_RGB_trainingdata

X_train_RGB, y_train_RGB = a_shuffle_RGB_trainingdata.shuffle_data(imgs_train_RGB, wheels_train_RGB)
# (4) 构建模型
import tensorflow as tf 
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda
from keras.layers import Conv2D, Dropout
from keras import losses 

# (5) 基准模型构建
# 选用nvidia model作为基准模型。
# NVIDIA model
import b_construct_nvidia_model

# (6) 模型训练
# Train the model with YUV data 

import b_train_RGB_part

construct_model = b_construct_nvidia_model.nvidia_model()

RGB_model, RGB_history = b_train_RGB_part.training(construct_model, X_train_RGB, y_train_RGB)

# (7) 在线推断
# Test the performance on test data

import c_model_inference

test_loss = c_model_inference.inference(RGB_model, imgs_test_RGB, wheels_test_RGB, batch_size)

import c_model_store

c_model_store.save_model(RGB_model)
# (8) 可视化结果比对
# viualize model

import c_visualize_direction_result

c_visualize_direction_result.visualize_model_label(RGB_history)

