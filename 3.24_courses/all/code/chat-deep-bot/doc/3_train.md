# Training options
* trainingArgs.add_argument('--numEpochs', type=int, default=30, help='maximum number of epochs to run')
* trainingArgs.add_argument('--saveEvery', type=int, default=2000, help='nb of mini-batch step before creating a model checkpoint')
* trainingArgs.add_argument('--batchSize', type=int, default=256, help='mini-batch size')
* trainingArgs.add_argument('--learningRate', type=float, default=0.002, help='Learning rate')
* trainingArgs.add_argument('--dropout', type=float, default=0.9, help='Dropout rate (keep probabilities)')

# 模型训练
可以添加其他参数
python .\main_a.py 


# 1 由训练数据创建模型需要batch输入

mainTrain/getBatches

1.1 确保数据的随机性
self.shuffle()
1.2 构造符合batch格式的每批次训练数据
(内部还能控制一定随机性)
textdata\_createBatch
函数内部也可以进行语句级别的打乱处理

# 2 训练 batch转换ops feedDict

mainTrain

2.1 将数据转换为model ops , 和输入数据
model / def step(self, batch):
ops, feedDict = self.model.step(nextBatch)

2.2 step中，会通过model构建了整体的网络结构

# 3 运行训练数据
_, loss, summary = sess.run(ops + (mergedSummaries,), feedDict)

