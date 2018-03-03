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

def frame_func(image):
    crop = image[300:650,500:,:]
    resized = cv2.resize(crop,(448,448))
    batch = np.array([resized[:,:,0],resized[:,:,1],resized[:,:,2]])
    batch = 2*(batch/255.) - 1
    batch = np.expand_dims(batch, axis=0)
    out = model.predict(batch)
    boxes = yolo_net_out_to_car_boxes(out[0], threshold = 0.17)
    return draw_box(boxes,image,[[500,1280],[300,650]])

def video_car_detection():
    project_video_output = 'E:/dl_data/yolo/project_video_output.mp4'
    clip1 = VideoFileClip("E:/dl_data/yolo/project_video.mp4")
    lane_clip = clip1.fl_image(frame_func) #NOTE: this function expects color images!!
    lane_clip.write_videofile(project_video_output, audio=False)
