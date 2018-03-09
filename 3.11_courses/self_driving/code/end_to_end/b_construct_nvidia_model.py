# 构建模型
import tensorflow as tf 
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda
from keras.layers import Conv2D, Dropout
from keras import losses 
import params, utils
import os

data_dir = params.data_dir
out_dir = params.out_dir
model_dir = params.model_dir
img_dir = os.path.abspath(params.data_dir + '/images/img/')
img_height = params.img_height
img_width = params.img_width
img_channels = params.img_channels
batch_size = params.batch_size
# 基准模型
# 选用nvidia model作为基准模型。
# NVIDIA model
def nvidia_model():
    import tensorflow as tf 
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Flatten, Lambda
    from keras.layers import Conv2D, Dropout
    from keras import losses 

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


# refined model
def refined_model():
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
    model.add(Dropout(0.3))
    model.add(Dense(100,kernel_initializer='he_normal',
                    activation='elu'))
    model.add(Dropout(0.3))
    model.add(Dense(50,kernel_initializer='he_normal',
                    activation='elu'))
    model.add(Dropout(0.3))
    model.add(Dense(10,kernel_initializer='he_normal',
                    activation='elu'))
    model.add(Dropout(0.4))
    
    model.add(Dense(1,kernel_initializer='he_normal'))
    
    model.compile(loss='mse', optimizer='Adadelta')
    
    return model

# final model
def final_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
             input_shape=(img_height, img_width, img_channels)))
    
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2,2), padding='valid',
                    kernel_initializer='he_normal', activation='elu'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2,2), padding='valid',
                    kernel_initializer='he_normal', activation='elu'))
    
    
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(1,1), padding='valid',
                    kernel_initializer='he_normal', activation='elu'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding='valid',
                    kernel_initializer='he_normal', activation='elu'))
    
    
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding='valid',
                    kernel_initializer='he_normal', activation='elu'))
    
    
    model.add(Flatten())
    model.add(Dense(1164, kernel_initializer='he_normal',
                    activation='elu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(100,kernel_initializer='he_normal',
                    activation='elu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(50,kernel_initializer='he_normal',
                    activation='elu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(10,kernel_initializer='he_normal',
                    activation='elu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(1,kernel_initializer='he_normal'))
    
    model.compile(loss='mse', optimizer='Adadelta')
    
    return model