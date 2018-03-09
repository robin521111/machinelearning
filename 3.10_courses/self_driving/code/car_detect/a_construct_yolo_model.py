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

def construct_yolo_model():
    # th input (channels, height, width)
    # tf input (height, width, channels)
    keras.backend.set_image_dim_ordering('th')
    model = Sequential()
    model.add(Convolution2D(16, 3, 3,input_shape=(3,448,448),border_mode='same',subsample=(1,1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    model.add(Convolution2D(64,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    model.add(Convolution2D(128,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    model.add(Convolution2D(256,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    model.add(Convolution2D(512,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    model.add(Convolution2D(1024,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Convolution2D(1024,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Convolution2D(1024,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dense(4096))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(1470))
    model.summary()
    return model


def construct_refine_model():
    keras.backend.set_image_dim_ordering('th')
    feature_layers = [
    Convolution2D(16, 3, 3,input_shape=(3,448,448),border_mode='same',subsample=(1,1)),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2)),
    Convolution2D(32,3,3 ,border_mode='same'),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2),border_mode='valid'),
    Convolution2D(64,3,3 ,border_mode='same'),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2),border_mode='valid'),
    Convolution2D(128,3,3 ,border_mode='same'),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2),border_mode='valid'),
    Convolution2D(256,3,3 ,border_mode='same'),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2),border_mode='valid'),
    Convolution2D(512,3,3 ,border_mode='same'),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2),border_mode='valid'),
    Convolution2D(1024,3,3 ,border_mode='same'),
    LeakyReLU(alpha=0.1),
    Convolution2D(1024,3,3 ,border_mode='same'),
    LeakyReLU(alpha=0.1),
    Convolution2D(1024,3,3 ,border_mode='same'),
    LeakyReLU(alpha=0.1),
    Flatten(),
    Dense(256),
    Dense(4096),
    LeakyReLU(alpha=0.1)
    ]

    classification_layers = [Dense(1470)]
    model = Sequential(feature_layers + classification_layers)
    # freeze feature layers and rebuild model
    for l in feature_layers:
        l.trainable = False
    for l in classification_layers:
        l.trainable = True
    model.summary()
    return model

def trainConstructModel():
    model = construct_refine_model()
    y_train = keras.utils.to_categorical(train[1], num_classes)
    y_test = keras.utils.to_categorical(test[1], num_classes)
    import b_load_model_weights

    b_load_model_weights.load_model_weights(model)

    # (1) 数据探索
    #  data explore
    import c_visualize_image

    resized = c_visualize_image.visualize_images()
    import d_inference_image
    out = d_inference_image.inference_image(model, resized)
    
    # 转换为Channel First
    batch = np.transpose(resized,(2,0,1))
    # 将图像转换到-1到1区间
    batch = 2*(batch/255.) - 1
    # 测试单张图片，需要将一个数据转换为数组batch数据
    batch = np.expand_dims(batch, axis=0)

    label = np.expand_dims(out, axis=0)

    model.compile(loss='mean_squared_error',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    t = now()
    model.fit(batch, label,
              batch_size=batch_size,
              epochs=1,
              verbose=1,
              validation_data=(x_test, y_test))

#construct_yolo_model()   