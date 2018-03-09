import tensorflow as tf 
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda
from keras.layers import Conv2D, Dropout
from keras import losses 
import time
import os
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

def training(model, X_train_RGB, y_train_RGB):
    RGB_model = model
    time_start = time.time()
    # Fit the model
    RGB_history = RGB_model.fit(X_train_RGB, y_train_RGB,
                              #epochs=8, 
                              epochs=5,
                              batch_size=batch_size)

    total_time = time.time() - time_start
    print('Training time: {}'.format(total_time))
    return RGB_model, RGB_history