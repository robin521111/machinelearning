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
img_dir = os.path.abspath(params.img_dir + '/img/')

img_height = params.img_height
img_width = params.img_width
img_channels = params.img_channels
batch_size = params.batch_size

# 加载数据

# load RGB data
import pickle

# load YUV data
pickle_file2 = 'data2.pickle'

with open ('data2.pickle', 'rb') as f:
    data = pickle.load(f)
    imgs_train_YUV = data['imgs_train_YUV']
    wheels_train_YUV = data['wheels_train_YUV'] 
    imgs_test_YUV = data['imgs_test_YUV']
    wheels_test_YUV = data['wheels_test_YUV']

print(imgs_train_YUV.shape, wheels_train_YUV.shape)
print(imgs_test_YUV.shape, wheels_test_YUV.shape)

# visualize the processed picture YUV
plt.imshow(imgs_train_YUV[0])
pylab.show()

# 数据分布
# 可视化Steering Angel的分布。
# 上图表明Steering Angle大致成正态分布，小幅转角出现的次数多，大幅转角出现的次数少。
wheel_sig = pd.read_csv(params.data_dir + '/epoch01_steering.csv')
wheel_sig.head()
wheel_sig.wheel.hist(bins=50)
# # 转角变化幅度
# # 可视化转角随时间的变化。
wheel_sig.plot(x='frame',y='wheel')
plt.show()

# 数据打乱处理
# 利用train_test_split函数将样本数据进行打乱处理，增加数据的随机性。
# shuffle train data 
from sklearn.model_selection import train_test_split

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
                              epochs=3,
                              batch_size=batch_size,
                              validation_data=(X_val_YUV, y_val_YUV))

total_time = time.time() - time_start
print('Training time: {}'.format(total_time))

# Test the performance on test data
test_loss= YUV_model.evaluate(imgs_test_YUV, wheels_test_YUV, batch_size=batch_size)

model_saved_path = os.path.join(params.model_dir, "model_YUV.h5")
json_saved_path = os.path.join(params.model_dir, "model_YUV.json")
# save model weight and json
json_model = YUV_model.to_json()
with open(json_saved_path, "w") as json_file:
    json_file.write(json_model)
# save model
# save model
YUV_model.save(model_saved_path)

# viualize model
print('Test loss is:{}'.format(test_loss))
print(YUV_history.history['loss'])
print(YUV_history.history['val_loss'])
print(YUV_history.history.keys())
plt.figure(figsize=(9,6))
plt.plot(YUV_history.history['loss'])
plt.plot(YUV_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.legend(['train YUV', 'valid YUV'], loc='upper right')
#plt.xlim((0,8))
#plt.xticks(np.arange(0, 8, 1))
plt.grid()
plt.savefig(img_dir + "\\nvidia_loss_compare.png", dpi=300)
plt.show()

# # 模型评估
# steering angle predicted by the model
machine_steering = YUV_model.predict(imgs_test_YUV, batch_size=128, verbose=0)

# steering angle controlled by human
human_steering = utils.get_human_steering(10)

# plot the steering angle predicted by human and by model
plt.figure
plt.plot(machine_steering)
plt.plot(human_steering)
plt.ylabel('Steering angle (deg)', fontsize=12)
plt.xlabel('Frame counts', fontsize=12)
plt.legend(['Machine steering', 'Human steering'], loc='upper right')
plt.xlim((0,2700))

plt.grid()
plt.savefig(img_dir + "\\result.png", dpi=300)
plt.show()
