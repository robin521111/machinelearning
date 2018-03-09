import pickle  # Saving the data

import os  # Checking file existance
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

def dumpBatchImage():
    from a_construct_yolo_model import construct_refine_model
    model = construct_refine_model()
    import b_load_model_weights
    b_load_model_weights.load_model_weights(model)
    # more examples
    images = [plt.imread(file) for file in glob.glob('./test_images/*.jpg')]
    batch = np.array([np.transpose(cv2.resize(image[300:650,500:,:],(448,448)),(2,0,1)) 
                  for image in images])
    batch = 2*(batch/255.) - 1
    out = model.predict(batch)

    with open(os.path.join("batch_data_6.pkl"), 'wb') as handle:
        data = {  # Warning: If adding something here, also modifying loadDataset
            'x': images,
            'y': out
        }
        pickle.dump(data, handle, -1)  # Using the highest protocol available

    with open("batch_data_6.pkl", 'rb') as handle:
        data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
        batch = data['x']
        label = data['y']
       
dumpBatchImage()
