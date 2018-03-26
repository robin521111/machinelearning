# 模型预测
python .\main_a.py --test interactive

# 1 输入单一一句话

singlePredict

# 2 训练 batch转换ops feedDict

mainTrain

2.1 将数据转换为model ops , 和输入数据
model / def step(self, batch):
ops, feedDict = self.model.step(nextBatch)

2.2 step中，会通过model构建了整体的网络结构DAG

# 3 运行训练数据
_, loss, summary = sess.run(ops + (mergedSummaries,), feedDict)