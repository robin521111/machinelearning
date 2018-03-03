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

def visualize_images():
    imagePath = './test_images/test1.jpg'
    image = plt.imread(imagePath)
    image_crop = image[300:650,500:,:]
    resized = cv2.resize(image_crop,(448,448))

    # apply the model to a test image
    f1,(ax11,ax22,ax33) = plt.subplots(1,3,figsize=(16,6))
    ax11.imshow(image)
    ax22.imshow(image_crop)
    ax33.imshow(resized)
    pylab.show()
    return resized 