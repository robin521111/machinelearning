# 对话机器人

#### Table of Contents

* [简介](#简介)
* [数据集](#数据集)
* [安装依赖](#安装依赖)
* [运行](#运行)
    * [Chatbot模型训练](#chatbot模型训练)
    * [Chatbot启动](#chatbot启动)
    * [Chatbot在线推断](#chatbot在线推断)
     * [Chatbot命令行](#chatbot命令行)
* [结果](#chatbot在线推断)
* [Pretrained model](#pretrained-model)

## 简介

## 数据集:
康奈尔数据集
 * [Cornell Movie Dialogs] (http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) corpus (default). Already included when cloning the repository.

The Cornell dataset is already included. 

To speedup the training, it's also possible to use pre-trained word embeddings (thanks to [Eschnou](https://github.com/eschnou)). More info [here](data/embeddings).

Here are some flags which could be useful. For more help and options, use `python main.py -h`:
 * `--playDataset`: show some dialogue samples from the dataset (can be use conjointly with `--createDataset` if this is the only action you want to perform).

## 安装依赖

### 1安装依赖
The program requires the following dependencies (easy to install using pip):
 * python 3.5
 * tensorflow (tested with v1.2)
 * numpy
 * nltk (natural language toolkit for tokenized the sentences)
 * tqdm (for the nice progression bars)
### 2下载依赖包
下载nltk数据集

```
python3 -m nltk.downloader punkt
```

## 运行


### Chatbot模型训练
执行训练模型
```cmd
 python .\main_a.py --modelTag train_model
```

To train the model, simply run `main.py`. Once trained, you can test the results with `main.py --test interactive` (more fun).

Here are some flags which could be useful. For more help and options, use `python main.py -h`:
 * `--modelTag <name>`: allow to give a name to the current model to differentiate between them when testing/training.
 * `--filterVocab 20` or `--vocabularySize 30000`: Limit the vocabulary size to and optimize the performances and memory usage. Replace the words used less than 20 times by the `<unknown>` token and set a maximum vocabulary size.
 * `--verbose`: when testing, will print the sentences as they are computed.

### Chatbot启动

1. 运行main.py启动bot framework后台服务
2. 启动bot framework emulator

Once trained, it's possible to chat with it using a more user friendly interface. The server will look at the model present on `save/model-server/model.ckpt`. The first time you want to use it, you'll need to configure it with:


### chatbot在线推断
命令行运行机器人
```cmd
 python .\main_a.py --test interactive
```

Surprisingly, it's possible to get some results after only 1 or 2 hours of training, by drastically reducing the input sequence to 5 words and output to 3 (plus the `go` and `eos` tokens) and by using a small embedding size (something like 25).
Of course, the network won't be really chatty:

    Q: Hi
    A: Hi.

    Q: What is your name ?
    A: Laura.

    Q: What does that mean ?
    A: I dunno.

    Q: How old are you ?
    A: thirty-five.

    Q: Will Google hire me ?
    A: No.

    Q: Tell me the alphabet
    A: Fuck you.

    Q: That's not nice
    A: Yeah.

Here is some case where it fails:

    Q: Who is Laura ?
    A: My brother.

    Q: Say 'goodbye'
    A: Alright.

    Q: What is cooking ?
    A: A channel.

    Q: Can you say no ?
    A: No.

    Q: Two plus two
    A: Manny...

### Chatbot命令行

Here are some flags which could be useful. For more help and options, use `python main.py -h`:

## Pretrained model

加载预训练模型
You can find a pre-trained model trained of the default corpus. To use it:
 1. Extract the zip file inside `/save/`
 2. Copy the preprocessed dataset from `save/model-pretrainedv2/dataset-cornell-old-lenght10-filter0-vocabSize0.pkl` to `data/samples/`.
 3. Run `./main.py --modelTag pretrainedv2 --test interactive`.

From my experiments, it seems that the learning rate and dropout rate have the most impact on the results. 


