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

def inference_and_visualize_batch_images_car_detection(model):
    # more examples
    images = [plt.imread(file) for file in glob.glob('./test_images/*.jpg')]
    batch = np.array([np.transpose(cv2.resize(image[300:650,500:,:],(448,448)),(2,0,1)) 
                  for image in images])
    batch = 2*(batch/255.) - 1
    out = model.predict(batch)
    f,((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,figsize=(11,10))
    th = 0.17
    for i,ax in zip(range(len(batch)),[ax1,ax2,ax3,ax4,ax5,ax6]):
    #boxes = yolo_net_out_to_car_boxes(out[i], threshold = 0.17)
        boxes = yolo_net_out_to_car_boxes(out[i], threshold = th)
        print("boxes" + str(i))
        print(boxes)
        ax.imshow(draw_box(boxes,images[i],[[500,1280],[300,650]]))
    pylab.show()