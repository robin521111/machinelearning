   # Network options 
   ## (Warning: if modifying something here, need retrain model, pretrained model not work )
每个RNN Cell有多少个隐藏单元？
* nnArgs.add_argument('--hiddenSize', type=int, default=512, help='number of hidden units in each RNN cell')
配置RNN layers层数
* nnArgs.add_argument('--numLayers', type=int, default=2, help='number of rnn layers')
Word2Vec维度
* nnArgs.add_argument('--embeddingSize', type=int, default=64, help='embedding size of the word representation')
反转模型输入数据，数据扩增
* globalArgs.add_argument('--watsonMode', action='store_true', help='Inverse the questions and answer when training (the network try to guess the question)')
随机使用Q&A对作为输入和输出
* globalArgs.add_argument('--autoEncode', action='store_true', help='Randomly pick the question or the answer and use it both as input and output')
