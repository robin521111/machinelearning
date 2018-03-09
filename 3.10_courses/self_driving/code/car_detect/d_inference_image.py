
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

def inference_image(model, resized):
    # 转换为Channel First
    batch = np.transpose(resized,(2,0,1))
    # 将图像转换到-1到1区间
    batch = 2*(batch/255.) - 1
    # 测试单张图片，需要将一个数据转换为数组batch数据
    batch = np.expand_dims(batch, axis=0)
    out = model.predict(batch)
    print("out results:")
    print(out)
    print("out shape")
    print(out.shape)
    return out