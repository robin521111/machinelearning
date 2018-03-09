import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import cv2
import pylab
np.random.seed(28)

# set the basic parameters
import params, utils

data_dir = params.data_dir
out_dir = params.out_dir
model_dir = params.model_dir
img_dir = os.path.abspath('./images/img/')

img_height = params.img_height
img_width = params.img_width
img_channels = params.img_channels
batch_size = params.batch_size

# 加载数据

# load RGB data
import pickle

pickle_file1 = 'data1.pickle'

with open ('data1.pickle', 'rb') as f:
    data = pickle.load(f)
    imgs_train_RGB = data['imgs_train_RGB']
    wheels_train_RGB = data['wheels_train_RGB']
    imgs_test_RGB = data['imgs_test_RGB']
    wheels_test_RGB = data['wheels_test_RGB']

# load YUV data
pickle_file2 = 'data2.pickle'

with open ('data2.pickle', 'rb') as f:
    data = pickle.load(f)
    imgs_train_YUV = data['imgs_train_YUV']
    wheels_train_YUV = data['wheels_train_YUV'] 
    imgs_test_YUV = data['imgs_test_YUV']
    wheels_test_YUV = data['wheels_test_YUV']


print(imgs_train_RGB.shape, wheels_train_RGB.shape)
print(imgs_train_YUV.shape, wheels_train_YUV.shape)
print(imgs_test_RGB.shape, wheels_test_RGB.shape)
print(imgs_test_YUV.shape, wheels_test_YUV.shape)


# visualize the processed picture RGB
plt.imshow(imgs_train_RGB[0])
pylab.show()
# visualize the processed picture YUV
plt.imshow(imgs_train_YUV[0])
pylab.show()

# 数据分布
# 可视化Steering Angel的分布。
# 上图表明Steering Angle大致成正态分布，小幅转角出现的次数多，大幅转角出现的次数少。
plt.figure
plt.hist(wheels_train_RGB,50);
plt.xlabel('Steering Angle')
plt.ylabel('Times')
#plt.title('Histogram of Steering Angle')
plt.grid(True)
plt.savefig(img_dir + '/angle_distribution.png')

# 转角变化幅度
# 可视化转角随时间的变化。
plt.figure
plt.plot(wheels_train_RGB[2000:5000])
plt.xlabel('time (s)')
plt.ylabel('Steering Angle (deg)')
#plt.title('Histogram of Steering Angle')
plt.grid(True)
plt.savefig(img_dir + '/Change_range.png')

# 数据打乱处理
# 利用train_test_split函数将样本数据进行打乱处理，增加数据的随机性。
# shuffle train data 
from sklearn.model_selection import train_test_split

# RGB mode
X_train_RGB, X_val_RGB, y_train_RGB, y_val_RGB = train_test_split(imgs_train_RGB, 
                                                                  wheels_train_RGB, train_size=0.8, random_state=28)
# YUV mode
X_train_YUV, X_val_YUV, y_train_YUV, y_val_YUV = train_test_split(imgs_train_YUV, 
                                                                  wheels_train_YUV, train_size=0.8, random_state=28)

# 构建模型
import tensorflow as tf 
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda
from keras.layers import Conv2D, Dropout
from keras import losses 

# 基准模型
# 选用nvidia model作为基准模型。
# NVIDIA model
def nvidia_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
             input_shape=(img_height, img_width, img_channels)))
    
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2,2), padding='valid',
                    kernel_initializer='he_normal', activation='elu'))
    
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2,2), padding='valid',
                    kernel_initializer='he_normal', activation='elu'))
    
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2,2), padding='valid',
                    kernel_initializer='he_normal', activation='elu'))
    
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding='valid',
                    kernel_initializer='he_normal', activation='elu'))
    
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding='valid',
                    kernel_initializer='he_normal', activation='elu'))
    
    model.add(Flatten())
    model.add(Dense(1164, kernel_initializer='he_normal',
                    activation='elu'))
    model.add(Dense(100,kernel_initializer='he_normal',
                    activation='elu'))
    model.add(Dense(50,kernel_initializer='he_normal',
                    activation='elu'))
    model.add(Dense(10,kernel_initializer='he_normal',
                    activation='elu'))
    
    model.add(Dense(1,kernel_initializer='he_normal'))
    
    model.compile(loss='mse', optimizer='Adadelta')
    return model

# Train the model with YUV data 
YUV_model = nvidia_model()
time_start = time.time()

# Fit the model
YUV_history = YUV_model.fit(X_train_YUV, y_train_YUV,
                              #epochs=8, 
                              epochs=1,
                              batch_size=batch_size,
                              validation_data=(X_val_YUV, y_val_YUV))

total_time = time.time() - time_start
print('Training time: {}'.format(total_time))

# Test the performance on test data
test_loss= YUV_model.evaluate(imgs_test_YUV, wheels_test_YUV, batch_size=batch_size)
print()
print('Test loss is:{}'.format(test_loss))
    

# Train the model with RGB data 
RGB_model = nvidia_model()
time_start = time.time()

# Fit the model
RGB_history = RGB_model.fit(X_train_RGB, y_train_RGB,
                              #epochs=8, 
                              epochs=1,
                              batch_size=batch_size,
                              validation_data=(X_val_RGB, y_val_RGB))

total_time = time.time() - time_start
print('Training time: {}'.format(total_time))

# Test the performance on test data
test_loss= RGB_model.evaluate(imgs_test_RGB, wheels_test_RGB, batch_size=batch_size)
print()
print('Test loss is:{}'.format(test_loss))

# summarize history for loss
#plt.plot(RGB_history.history['loss'], 'C0--')
plt.plot(RGB_history.history['loss'])
#plt.plot(RGB_history.history['val_loss'], 'C0')
plt.plot(RGB_history.history['val_loss'])

#plt.plot(YUV_history.history['loss'], 'C1--')
plt.plot(YUV_history.history['loss'])
#plt.plot(YUV_history.history['val_loss'], 'C1')
plt.plot(YUV_history.history['val_loss'])

# plt.title('model loss')
plt.ylabel('Loss', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.legend(['train RGB', 'valid RGB', 'train YUV', 'valid YUV'], loc='upper right')
plt.xlim((0,8))
plt.xticks(np.arange(0, 8, 1))
plt.grid()
plt.savefig(img_dir + "/nvidia_loss_compare.png", dpi=300)
plt.show()

# 改良模型
# refined model
# def refined_model():
#     model = Sequential()
#     model.add(Lambda(lambda x: x/127.5 - 1.,
#              input_shape=(img_height, img_width, img_channels)))
    
#     model.add(Conv2D(24, kernel_size=(5, 5), strides=(2,2), padding='valid',
#                     kernel_initializer='he_normal', activation='elu'))
    
#     model.add(Conv2D(36, kernel_size=(5, 5), strides=(2,2), padding='valid',
#                     kernel_initializer='he_normal', activation='elu'))
    
#     model.add(Conv2D(48, kernel_size=(5, 5), strides=(2,2), padding='valid',
#                     kernel_initializer='he_normal', activation='elu'))
    
#     model.add(Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding='valid',
#                     kernel_initializer='he_normal', activation='elu'))
    
#     model.add(Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding='valid',
#                     kernel_initializer='he_normal', activation='elu'))
    
#     model.add(Flatten())
#     model.add(Dense(1164, kernel_initializer='he_normal',
#                     activation='elu'))
#     model.add(Dropout(0.3))
#     model.add(Dense(100,kernel_initializer='he_normal',
#                     activation='elu'))
#     model.add(Dropout(0.3))
#     model.add(Dense(50,kernel_initializer='he_normal',
#                     activation='elu'))
#     model.add(Dropout(0.3))
#     model.add(Dense(10,kernel_initializer='he_normal',
#                     activation='elu'))
#     model.add(Dropout(0.4))
    
#     model.add(Dense(1,kernel_initializer='he_normal'))
    
#     model.compile(loss='mse', optimizer='Adadelta')
    
#     return model

# # Train the model with RGB_refined data 
# RGB_refined_model = refined_model()
# time_start = time.time()

# # Fit the model
# RGB_refined_history = RGB_refined_model.fit(X_train_RGB, y_train_RGB,
#                               epochs=10, 
#                               batch_size=batch_size,
#                               validation_data=(X_val_RGB, y_val_RGB))

# total_time = time.time() - time_start
# print('Training time: {}'.format(total_time))

# # Test the performance on test data
# test_loss= RGB_refined_model.evaluate(imgs_test_RGB, wheels_test_RGB, batch_size=batch_size)
# print()
# print('Test loss is:{}'.format(test_loss))

# # Train the model with YUV_refined data 
# YUV_refined_model = refined_model()
# time_start = time.time()

# # Fit the model
# YUV_refined_history = YUV_refined_model.fit(X_train_YUV, y_train_YUV,
#                               epochs=10, 
#                               batch_size=batch_size,
#                               validation_data=(X_val_YUV, y_val_YUV))

# total_time = time.time() - time_start
# print('Training time: {}'.format(total_time))

# # Test the performance on test data
# test_loss= YUV_refined_model.evaluate(imgs_test_YUV, wheels_test_YUV, batch_size=batch_size)
# print()
# print('Test loss is:{}'.format(test_loss))

# # summarize history for loss
# plt.plot(RGB_refined_history.history['loss'], 'C0--')
# plt.plot(RGB_refined_history.history['val_loss'], 'C0')

# plt.plot(YUV_refined_history.history['loss'], 'C1--')
# plt.plot(YUV_refined_history.history['val_loss'], 'C1')

# # plt.title('model loss')
# plt.ylabel('Loss', fontsize=12)
# plt.xlabel('Epoch', fontsize=12)
# plt.legend(['train RGB', 'valid RGB', 'train YUV', 'valid YUV'], loc='upper right')
# plt.xlim((0,10))
# plt.xticks(np.arange(0, 10, 1))
# plt.grid()
# plt.savefig(img_dir + "/refined_loss_compare.png", dpi=300)
# plt.show()

# # 最终模型
# from keras.layers import MaxPooling2D
# from keras import regularizers
# from keras.layers.normalization import BatchNormalization

# # final model
# def final_model():
#     model = Sequential()
#     model.add(Lambda(lambda x: x/127.5 - 1.,
#              input_shape=(img_height, img_width, img_channels)))
    
#     model.add(Conv2D(24, kernel_size=(5, 5), strides=(2,2), padding='valid',
#                     kernel_initializer='he_normal', activation='elu'))
#     model.add(BatchNormalization())
    
#     model.add(Conv2D(36, kernel_size=(5, 5), strides=(2,2), padding='valid',
#                     kernel_initializer='he_normal', activation='elu'))
    
    
#     model.add(Conv2D(48, kernel_size=(5, 5), strides=(1,1), padding='valid',
#                     kernel_initializer='he_normal', activation='elu'))
#     model.add(BatchNormalization())
    
#     model.add(Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding='valid',
#                     kernel_initializer='he_normal', activation='elu'))
    
    
#     model.add(Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding='valid',
#                     kernel_initializer='he_normal', activation='elu'))
    
    
#     model.add(Flatten())
#     model.add(Dense(1164, kernel_initializer='he_normal',
#                     activation='elu'))
#     model.add(Dropout(0.2))
    
#     model.add(Dense(100,kernel_initializer='he_normal',
#                     activation='elu'))
#     model.add(Dropout(0.2))
    
#     model.add(Dense(50,kernel_initializer='he_normal',
#                     activation='elu'))
#     model.add(Dropout(0.3))
    
#     model.add(Dense(10,kernel_initializer='he_normal',
#                     activation='elu'))
#     model.add(Dropout(0.3))
    
#     model.add(Dense(1,kernel_initializer='he_normal'))
    
#     model.compile(loss='mse', optimizer='Adadelta')
    
#     return model

# # Train the model with RGB_final data 
# RGB_final_model = final_model()
# time_start = time.time()

# # Fit the model
# RGB_final_history = RGB_final_model.fit(X_train_RGB, y_train_RGB,
#                               epochs=8, 
#                               batch_size=batch_size,
#                               validation_data=(X_val_RGB, y_val_RGB))

# total_time = time.time() - time_start
# print('Training time: {}'.format(total_time))

# # Test the performance on test data
# test_loss= RGB_final_model.evaluate(imgs_test_RGB, wheels_test_RGB, batch_size=batch_size)
# print()
# print('Test loss is:{}'.format(test_loss))

# # Train the model with YUV_final data 
# YUV_final_model = final_model()
# time_start = time.time()

# # Fit the model
# YUV_final_history = YUV_final_model.fit(X_train_YUV, y_train_YUV,
#                               epochs=8, 
#                               batch_size=batch_size,
#                               validation_data=(X_val_YUV, y_val_YUV))

# total_time = time.time() - time_start
# print('Training time: {}'.format(total_time))

# # Test the performance on test data
# test_loss= YUV_final_model.evaluate(imgs_test_YUV, wheels_test_YUV, batch_size=batch_size)
# print()
# print('Test loss is:{}'.format(test_loss))

# # summarize history for loss
# plt.plot(RGB_final_history.history['loss'], 'C0--')
# plt.plot(RGB_final_history.history['val_loss'], 'C0')

# plt.plot(YUV_final_history.history['loss'], 'C1--')
# plt.plot(YUV_final_history.history['val_loss'], 'C1')

# # plt.title('model loss')
# plt.ylabel('Loss', fontsize=12)
# plt.xlabel('Epoch', fontsize=12)
# plt.legend(['train RGB', 'valid RGB', 'train YUV', 'valid YUV'], loc='upper right')
# plt.xlim((0,8))
# plt.xticks(np.arange(0, 8, 1))
# plt.grid()
# plt.savefig(img_dir + "/final_loss_compare.png", dpi=300)
# plt.show()

# # model and json save path
# model_saved_path = os.path.join(params.model_dir, "model.h5")
# json_saved_path = os.path.join(params.model_dir, "model.json")

# # save json
# json_model = RGB_final_model.to_json()
# with open(json_saved_path, "w") as json_file:
#     json_file.write(json_model)
# # save model
# RGB_final_model.save(model_saved_path)

# # 模型结构
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot

# SVG(model_to_dot(RGB_final_model, show_shapes=True).create(prog='dot', format='svg'))

# # 模型评估
# # steering angle predicted by the model
# machine_steering = RGB_final_model.predict(imgs_test_YUV, batch_size=128, verbose=0)

# # steering angle controlled by human
# human_steering = utils.get_human_steering(10)

# # plot the steering angle predicted by human and by model
# plt.figure
# plt.plot(machine_steering)
# plt.plot(human_steering)
# plt.ylabel('Steering angle (deg)', fontsize=12)
# plt.xlabel('Frame counts', fontsize=12)
# plt.legend(['Machine steering', 'Human steering'], loc='upper right')
# plt.xlim((0,2700))

# plt.grid()
# plt.savefig(img_dir + "/result.png", dpi=300)
# plt.show()