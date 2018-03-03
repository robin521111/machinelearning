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

def visualize_image_car_detection(boxes):
    imagePath = './test_images/test1.jpg'
    image = plt.imread(imagePath)
    print("boxes results:")
    print(boxes)
    # visualize the box on the original image
    f,(ax1,ax2) = plt.subplots(1,2,figsize=(16,6))
    ax1.imshow(image)
    ax2.imshow(draw_box(boxes,plt.imread(imagePath),[[500,1280],[300,650]]))
    pylab.show() 