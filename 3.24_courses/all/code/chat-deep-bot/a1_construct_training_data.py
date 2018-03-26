import os
import re
from tqdm import tqdm
import io
import sys
import numpy as np
import nltk  # For tokenize
from tqdm import tqdm  # Progress bar
import pickle  # Saving the data
import math  # For float comparison
import os  # Checking file existance
import random
import string
import collections
from chatbot.corpus.cornelldata import CornellData
from chatbot.corpus.xiaohuangdata import XiaohuangData
#ys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')


def tqdm_wrap(iterable, *args, **kwargs):
    """Forward an iterable eventually wrapped around a tqdm decorator
    The iterable is only wrapped if the iterable contains enough elements
    Args:
        iterable (list): An iterable object which define the __len__ method
        *args, **kwargs: the tqdm parameters
    Return:
        iter: The iterable eventually decorated
    """
    if len(iterable) > 100:
        return tqdm(iterable, *args, **kwargs)
    return iterable

maxLength = 10
filterVocab = 0
vocabularySize = 1000
## save data to pickle file
#modelTag = "xiaohuang"
#corpus = "xiaohuang"
modelTag = "cornell"
corpus = "cornell"
basePath="data/samples/dataset-" + modelTag
fullSamplesPath = basePath + '.pkl'  # Full sentences length/vocab
filteredSamplesPath = basePath + '-length{}-filter{}-vocabSize{}.pkl'.format(
    maxLength,
    filterVocab,
    vocabularySize,
)  # Sentences/vocab filtered for this model

# padToken = -1  # Padding
# goToken = -1  # Start of sequence
# eosToken = -1  # End of sequence
# unknownToken = -1  # Word dropped from vocabulary
padToken = 0  # Padding
goToken = 0  # Start of sequence
eosToken = 0  # End of sequence
unknownToken = 0  # Word dropped from vocabulary
# 2d array containing each question and his answer [[input,target]]
trainingSamples = []
word2id = {} # "你": 1
# For a rapid conversion (Warning: If replace dict by list, modify the filtering to avoid linear complexity with del)
id2word = {} # 1 : "你"
# Useful to filters the words (TODO: Could replace dict by list or use collections.Counter)
idCount = {}

# 1 text > lines
def getLinesFromXiaohuang():
    conversations = []
    root_dir = "data/xiaohuang"
    qfile = open(root_dir + "/question.txt", 'r', encoding="utf-8")
    afile = open(root_dir + "/answer.txt", 'r', encoding="utf-8")
    q1 = ""
    a1 = ""
    for q in qfile:
        a = str(afile.readline())
        lines = []
        lines.append({"text": q})
        lines.append({"text": a})
        conversations.append({"lines": lines})
    # [{"lines": [{"text": "你好"}, {"text": "你好啊"}, {}]}, {}]
    qfile.close()
    afile.close()
    return conversations

#print(getLinesFromXiaohuang())
# 2 lines > pickle
def getWordId(word, create=True):
    # Should we Keep only words with more than one occurrence ?
    word = word.lower()  # Ignore case
    # At inference, we simply look up for the word
    if not create:
        wordId = word2id.get(word, unknownToken)
    # Get the id if the word already exist
    elif word in word2id:
        wordId = word2id[word]
        idCount[wordId] += 1
    # If not, we create a new entry
    else:
        wordId = len(word2id)
        word2id[word] = wordId
        id2word[wordId] = word
        idCount[wordId] = 1
    return wordId

def printCollection():
    print("training samples")
    print(trainingSamples[0:5])
    print("word2id")
    print(word2id)
    print("id2word")
    print(id2word)
    print("idCount")
    print(idCount)

def extractConversation(conversation):
    # WARNING: The dataset won't be regenerated if the choice evolve (have to use the datasetTag)
    step = 1
 
    # Iterate over all the lines of the conversation
    for i in tqdm_wrap(
        range(0, len(conversation['lines']) - 1, step),  # We ignore the last line (no answer for it)
        desc='Conversation',
        leave=False
    ):
        # 将对话按时间顺序组合成Q&A. step决定滑动的步长是多少
        inputLine  = conversation['lines'][i] # x
        targetLine = conversation['lines'][i+1] # y
        print(str(inputLine))
        inputWords  = extractText(inputLine['text']) # 你好啊 -> [2, 3, 6]
        targetWords = extractText(targetLine['text'])

        if inputWords and targetWords:  # Filter wrong samples (if one of the list is empty)
            trainingSamples.append([inputWords, targetWords])

# 句子分词，转换为句子向量
def extractText(line):
    """Extract the words from a sample lines
    Args:
        line (str): a line containing the text to extract
    Return:
        list<list<int>>: the list of sentences of word ids of the sentence
    """
    sentences = []  # List[List[str]]
    tokens = []
    # Extract sentences
    sentencesToken = nltk.sent_tokenize(line)
    # We add sentence by sentence until we reach the maximum length
    for i in range(len(sentencesToken)): 
        if corpus == "xiaohuang":
            tokens = list(line) # 你好啊， [你， 好， 啊], jieba
        else:
            tokens = nltk.word_tokenize(sentencesToken[i])

        tempWords = []
        index = 0
        for token in tokens:
            if index % 100 == 0:
                print(token)
            index += 1
            tempWords.append(getWordId(token))  # Create the vocabulary and the training sentences
        sentences.append(tempWords)

    return sentences # [1, 4, 5]


def saveDataset(filename):
    """Save samples to file
    Args:
        filename (str): pickle filename
    """
    with open(os.path.join(filename), 'wb') as handle:
        data = {  # Warning: If adding something here, also modifying loadDataset
            'word2id': word2id,
            'id2word': id2word,
            'idCount': idCount,
            'trainingSamples': trainingSamples
        }
        pickle.dump(data, handle, -1)  # Using the highest protocol available

# 加载数据之前预处理数据，然后进行一定预处理
def filterFromFull():
    """ Load the pre-processed full corpus and filter the vocabulary / sentences
    to match the given model options
    """
    # 对Q或者A，如果数据长度超过maxlength,则停止，如果不超过则合并词到句子里，可以控制是
    # 从前往后也可以从后往前进行截取
    def mergeSentences(sentences, fromEnd=False):
        """Merge the sentences until the max sentence length is reached
        Also decrement id count for unused sentences.
        Args:
            sentences (list<list<int>>): the list of sentences for the current line
            fromEnd (bool): Define the question on the answer
        Return:
            list<int>: the list of the word ids of the sentence
        """
        # We add sentence by sentence until we reach the maximum length
        merged = []

        # If question: we only keep the last sentences
        # If answer: we only keep the first sentences
        if fromEnd:
            sentences = reversed(sentences)

        for sentence in sentences:

            # If the total length is not too big, we still can add one more sentence
            if len(merged) + len(sentence) <= maxLength:
                if fromEnd:  # Append the sentence
                    merged = sentence + merged
                else:
                    merged = merged + sentence
            else:  # If the sentence is not used, neither are the words
                for w in sentence:
                    idCount[w] -= 1
        return merged

    newSamples = []
    # 1 根据指定句子词长度进行过滤
    # 1st step: Iterate over all words and add filters the sentences
    # according to the sentence lengths
    for inputWords, targetWords in tqdm(trainingSamples, desc='Filter sentences:', leave=False):
        inputWords = mergeSentences(inputWords, fromEnd=True)
        targetWords = mergeSentences(targetWords, fromEnd=False)
        newSamples.append([inputWords, targetWords])
    words = []

    # WARNING: DO NOT FILTER THE UNKNOWN TOKEN !!! Only word which has count==0 ?
    # 2 过滤低频词
    # 2nd step: filter the unused words and replace them by the unknown token
    # This is also where we update the correnspondance dictionaries
    specialTokens = {  # TODO: bad HACK to filter the special tokens. Error prone if one day add new special tokens
        padToken,
        goToken,
        eosToken,
        unknownToken
    }
    newMapping = {}  # Map the full words ids to the new one (TODO: Should be a list)
    newId = 0

    selectedWordIds = collections \
        .Counter(idCount) \
        .most_common(vocabularySize or None)  # Keep all if vocabularySize == 0
    selectedWordIds = {k for k, v in selectedWordIds if v > filterVocab}
    selectedWordIds |= specialTokens

    for wordId, count in [(i, idCount[i]) for i in range(len(idCount))]:  # Iterate in order
        if wordId in selectedWordIds:  # Update the word id
            newMapping[wordId] = newId
            word = id2word[wordId]  # The new id has changed, update the dictionaries
            del id2word[wordId]  # Will be recreated if newId == wordId
            word2id[word] = newId
            id2word[newId] = word
            newId += 1
        else:  # Cadidate to filtering, map it to unknownToken (Warning: don't filter special token)
            newMapping[wordId] = unknownToken
            del word2id[id2word[wordId]]  # The word isn't used anymore
            del id2word[wordId]
    # 3 更新词频id
    # Last step: replace old ids by new ones and filters empty sentences
    def replace_words(words):
        valid = False  # Filter empty sequences
        for i, w in enumerate(words):
            words[i] = newMapping[w]
            if words[i] != unknownToken:  # Also filter if only contains unknown tokens
                valid = True
        return valid

    trainingSamples.clear()

    for inputWords, targetWords in tqdm(newSamples, desc='Replace ids:', leave=False):
        valid = True
        valid &= replace_words(inputWords)
        valid &= replace_words(targetWords)
        #valid &= targetWords.count(self.unknownToken) == 0  # Filter target with out-of-vocabulary target words ?

        if valid:
            trainingSamples.append([inputWords, targetWords])  # TODO: Could replace list by tuple

    idCount.clear()  # Not usefull anymore. Free data

# Add standard tokens
# Padding (Warning: first things to add > id=0 !!)
padToken = getWordId('<pad>')
goToken = getWordId('<go>')  # Start of sequence
eosToken = getWordId('<eos>')  # End of sequence
unknownToken = getWordId('<unknown>')  # Word dropped from vocabulary

print(padToken)
print(goToken)
print(eosToken)
print(unknownToken)

def main():
    # Preprocessing data
    # 1 抽取对话
    import chatbot
    cornel = chatbot.corpus.cornelldata.CornellData("data/cornell")
    conversations = cornel.getConversations()
    #conversations = getLinesFromXiaohuang()
    for conversation in tqdm(conversations, desc='Extract conversations'):
        # 2 分词，构建词典，向量化语句
        extractConversation(conversation)
    # 3 过滤，规范化
    filterFromFull()
    # 4 保存数据
    saveDataset(filteredSamplesPath)
    printCollection()
    #print(trainingSamples[0:5])
    print(filteredSamplesPath)

main()

# 3 loadPickle > batches
# 4 batches > step input
# 5 train
# 6 line -> ids
# 7 ids predict -> ids
# 8 ids -> line


