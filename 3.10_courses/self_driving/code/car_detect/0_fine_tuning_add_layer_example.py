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

def trainConstructModel():
    from a_construct_yolo_model import construct_refine_model
    model = construct_refine_model()

    import b_load_model_weights

    b_load_model_weights.load_model_weights(model)

    # # (1) 数据探索
    # #  data explore
    # imagePath = './test_images/test1.jpg'
    # image = plt.imread(imagePath)
    # image_crop = image[300:650,500:,:]
    # resized = cv2.resize(image_crop,(448,448))

    # import d_inference_image
    # out = d_inference_image.inference_image(model, resized)

    # # 转换为Channel First
    # batch = np.transpose(resized,(2,0,1))
    # # 将图像转换到-1到1区间
    # batch = 2*(batch/255.) - 1
    # # 测试单张图片，需要将一个数据转换为数组batch数据
    # batch = np.expand_dims(batch, axis=0)

    # label = np.expand_dims(out[0], axis=0)

    model.add(Dense(1470))
    # model.compile(loss='mean_squared_error',
    #               optimizer='adadelta',
    #               metrics=['accuracy'])

    # model.fit(batch, label,
    #           batch_size=1,
    #           nb_epoch= 1, # other is 13 is ok
    #           verbose=1,
    #           validation_data=(batch, label))

    # # model inference the image
    # import d_inference_image

    # out = d_inference_image.inference_image(model, resized)

    # # (3) YOLO模型预测结果转换为box坐标
    # # ineference result convert to box
    # import e_convert_to_box

    # boxes = e_convert_to_box.regression_convert_to_box(out)

    # # (4) 单图像上进行车辆检测
    # # visualize car detection image results
    # import f_visualize_image_car_detection

    # f_visualize_image_car_detection.visualize_image_car_detection(boxes)

trainConstructModel()