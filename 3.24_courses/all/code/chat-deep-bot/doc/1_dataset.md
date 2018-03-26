  # Dataset options 训练数据集控制
生成数据集
* globalArgs.add_argument('--createDataset', action='store_true', help='if present, the program will only generate the dataset from the corpus (no training/testing)')
数据摸底
* globalArgs.add_argument('--playDataset', type=int, nargs='?', const=10, default=None,  help='if set, the program  will randomly play some samples(can be use conjointly with createDataset if this is the only action you want to perform)')
数据集选用，默认使用康奈尔数据集
* datasetArgs.add_argument('--corpus', choices=TextData.corpusChoices(), default=TextData.corpusChoices()[0], help='corpus on which extract the dataset.')

数据集标签控制
* datasetArgs.add_argument('--datasetTag', type=str, default='', help='add a tag to the dataset (file where to load the vocabulary and the precomputed samples, not the original corpus). Useful to manage multiple versions. Also used to define the file used for the lightweight format.')  # The samples are computed from the corpus if it does not exist already. There are saved in \'data/samples/\'

训练数据集使用比例控制
* datasetArgs.add_argument('--ratioDataset', type=float, default=1.0, help='ratio of dataset used to avoid using the whole dataset')  # Not implemented, useless ?

RNN输入输出句子长度控制
* datasetArgs.add_argument('--maxLength', type=int, default=10, help='maximum length of the sentence (for input and output), define number of maximum step of the RNN')

低频词过滤

* datasetArgs.add_argument('--filterVocab', type=int, default=1, help='remove rarelly used words (by default words used only once). 0 to keep all words.')
* datasetArgs.add_argument('--skipLines', action='store_true', help='Generate training samples by only using even conversation lines as questions (and odd lines as answer). Useful to train the network on a particular person.')

词典数量控制
* datasetArgs.add_argument('--vocabularySize', type=int, default=40000, help='Limit the number of words in the vocabulary (0 for unlimited)')


# 整体流程
## 1. 抽取对话（语料-》Q&A句子）
语料抽取出q&a对
1.1 corpus/cornelldata.py 以会话为单位，会话-》对话时序，包含角色信息
1.2 textdata/createFullCorpus/extractConversation 将数据存储为q&a对，并将词典话语句转储为二进制数据
## 2. 分词（句子-》词）
分词
textdata/createFullCorpus/extractText/nltk.tokenize
## 3. 构建词典（词-》id）
构建词典
textdata/createFullCorpus/extractText/getWordId
## 4. 句子向量化（句子-》id向量）
textdata/createFullCorpus/extractText/

同时：对话时序 顺序存储为Q&A对，作为x和y值，step决定滑动的步长
## 5. 保存预处理好的数据集和字典

textdata/saveDataset

'word2id': self.word2id,
'id2word': self.id2word,
'idCount': self.idCount,
'trainingSamples': self.trainingSamples

## 6. 加载使用数据集
loadDataset
## 7. 预测结果数字列表转换为词串
sequence2str
词串转换为语句

## 8. 全流程测试

### 8.1 句子转换为id串作为model输入
sentence2enco
### 8.2 将decoder输出的id串转换为句子
deco2sentence

## 9. 命令：数据摸底，打印样本数据
playDataset

这个命令和其他联合执行
python .\main_a.py --playDataset --test

* `--playDataset`: show some dialogue samples from the dataset (can be use conjointly with `--createDataset` if this is the only action you want to perform).

## 10. 由原始语料创建训练数据集
text -> (input， output) id串 等

10.1 数据存储与加载路径
判断路径下是否存在相应的数据
如果过滤不了则使用全量数据,数据存储在sample路径下
path = os.path.join(self.args.rootDir, 'data/samples/')
self.fullSamplesPath = basePath + '.pkl'  # Full sentences length/vocab
self.filteredSamplesPath = basePath + '-length{}-filter{}-vocabSize{}.pkl'.format(
            self.args.maxLength,
            self.args.filterVocab,
            self.args.vocabularySize,
        ) 

10.2  创建数据集
如果不存在则进行创建

python .\main_a.py --createDataset

测试：
python .\main_a.py --createDataset --maxLength 3 --vocabularySize 1000 --filterVocab 3
如果想调整则可以换参数或者拷贝走文件

python .\main_a.py --createDataset --maxLength 100 --vocabularySize 3000 --filterVocab 3 --corpus xiaohuang --embeddingSize 128
 python .\main_a.py --modelTag xiaohuang --numEpochs 1 --corpus xiaohuang --maxLength 3 --vocabularySize 1000 --filterVocab 3