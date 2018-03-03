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

def regression_convert_to_box(out):
    # interpolate the vector out from the neural network, generate the boxes
    #boxes = yolo_net_out_to_car_boxes(out[0], threshold = 0.17)
    th = 0.17
    boxes = yolo_net_out_to_car_boxes(out[0], threshold = th)
    return boxes