#encoding:utf-8
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D
from keras.optimizers import SGD
from keras.utils import to_categorical

(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#theano为后端的图片矩阵通道数在前，相反tensorflow是通道数在后
x_train=x_train.reshape(60000,28,28, 1)
x_test=x_test.reshape(10000,28,28, 1)

#把标签变成one-hot编码的形式
y_train=to_categorical(y_train,num_classes=10)
y_test=to_categorical(y_test,num_classes=10)

# create model
model=Sequential()
model.add(Conv2D(filters=6, kernel_size=(5,5), padding='valid', input_shape=(1,28,28), activation='tanh'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=16, kernel_size=(5,5), padding='valid', activation='tanh'))

model.add(MaxPool2D(pool_size=(2,2)))

#池化后变成16个4x4的矩阵，然后把矩阵压平变成一维的，一共256个单元。
model.add(Flatten())

#下面就是全连接层了
model.add(Dense(120, activation='tanh'))

model.add(Dense(84, activation='tanh'))

model.add(Dense(10, activation='softmax'))
#compile model

#事实证明，对于分类问题，使用交叉熵(cross entropy)作为损失函数更好些
model.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(lr=0.1),
    metrics=['accuracy']
)

#train model
model.fit(x_train,y_train,batch_size=128,epochs=2)

#evaluate model

score=model.evaluate(x_test,y_test)
print("Total loss on Testing Set:", score[0])
print("Accuracy of Testing Set:", score[1])