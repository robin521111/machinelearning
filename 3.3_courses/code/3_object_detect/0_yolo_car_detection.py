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
import a_construct_yolo_model

model = a_construct_yolo_model.construct_yolo_model()

import b_load_model_weights

b_load_model_weights.load_model_weights(model)

# (1) 数据探索
#  data explore
import c_visualize_image

resized = c_visualize_image.visualize_images()

# (2) 图像模型推断
# model inference the image
import d_inference_image

out = d_inference_image.inference_image(model, resized)

# (3) YOLO模型预测结果转换为box坐标
# ineference result convert to box
import e_convert_to_box

boxes = e_convert_to_box.regression_convert_to_box(out)

# (4) 单图像上进行车辆检测
# visualize car detection image results
import f_visualize_image_car_detection

f_visualize_image_car_detection.visualize_image_car_detection(boxes)

# (5) 可视化并对多个示例图像进行车辆检测
# Visualize more examples
import g_inference_batch_image

g_inference_batch_image.inference_and_visualize_batch_images_car_detection(model)
  
# (6) 对视频进行车辆检测
# apply to video
import h_video_car_detection

h_video_car_detection.video_car_detection()

