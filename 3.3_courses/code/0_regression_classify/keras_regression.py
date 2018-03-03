import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# 构建数据集
X_data = np.linspace(-1,1,100)[:, np.newaxis]
noise = np.random.normal(0,0.05,X_data.shape)
y_data = np.square(X_data) + noise + 0.5

# 构建神经网络
model = Sequential()
model.add(Dense(10, input_dim=1, init='normal', activation='relu'))
# vs 分类为softmax激活
model.add(Dense(1, init='normal'))
sgd = SGD(lr=0.1)
model.compile(loss='mean_squared_error', optimizer=sgd)
# 训练
model.fit(X_data, y_data, nb_epoch=1000, batch_size=100, verbose=0)
# 在原数据上预测
y_predict=model.predict(X_data,batch_size=100,verbose=1)

# 可视化
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(X_data, y_data)
ax.plot(X_data,y_predict,'r-',lw=5)
plt.show()