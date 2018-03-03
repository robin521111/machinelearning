import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import keras ## broken for keras >= 2.0, use 1.2.2
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Activation, Reshape
from utils import load_weights, Box, yolo_net_out_to_car_boxes, draw_box
import pylab 

# (0) 构建YOLO实时检测模型
# construct the tiny-yolo model 

# (1) 数据探索
#  data explore

# (2) 图像模型推断
# model inference the image

# (3) YOLO模型预测结果转换为box坐标
# ineference result convert to box

# (4) 单图像上进行车辆检测
# visualize car detection image results

# (5) 可视化并对多个示例图像进行车辆检测
# Visualize more examples
  
# (6) 对视频进行车辆检测
# apply to video

