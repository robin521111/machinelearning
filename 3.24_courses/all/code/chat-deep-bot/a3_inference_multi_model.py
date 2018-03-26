import argparse  # Command line parsing
import configparser  # Saving the models parameters
import argparse  # Command line parsing
import configparser  # Saving the models parameters
import datetime  # Chronometer
import os  # Files management
import tensorflow as tf
import numpy as np
import math
from tqdm import tqdm  # Progress bar
from tensorflow.python import debug as tf_debug
from chatbot.textdata import TextData
from chatbot.model import Model
from a0_args_util import ArgsUtil


class InferenceObject:
    # Filename and directories constants
    MODEL_DIR_BASE = 'save/model'
    MODEL_NAME_BASE = 'model'
    MODEL_EXT = '.ckpt'
    CONFIG_FILENAME = 'params.ini'
    CONFIG_VERSION = '0.5'
    TEST_IN_NAME = 'data/test/samples.txt'
    TEST_OUT_SUFFIX = '_predictions.txt'
    SENTENCES_PREFIX = ['Q: ', 'A: ']

    args = None
    # Task specific object
    textData = None  # Dataset
    model = None  # Sequence to sequence model
    # Tensorflow utilities for convenience saving/logging
    writer = None
    saver = None
    modelDir = MODEL_DIR_BASE  # Where the model is saved
    globStep = 0  # Represent the number of iteration for the current model
    sess = None

    def singlePredict(self, question, questionSeq=None):
        # Create the input batch
        batch = self.textData.sentence2enco(question)
        if not batch:
            return None
        if questionSeq is not None:  # If the caller want to have the real input
            questionSeq.extend(batch.encoderSeqs)
        # Run the model
        ops, feedDict = self.model.step(batch)
        output = self.sess.run(ops[0], feedDict)  # TODO: Summarize the output too (histogram, ...)
        #print("output" + str(output))
        #print(output)
        answer = self.textData.deco2sentence(output)
        print("answer: " + str(answer))
        return answer

    def daemonPredict(self, sentence):
        return self.textData.sequence2str(
            self.singlePredict(sentence),
            clean=True
    )

    def singlePredictMultiModel(self, question, sess, questionSeq=None):
        # Create the input batch
        batch = self.textData.sentence2enco(question)
        if not batch:
            return None
        if questionSeq is not None:  # If the caller want to have the real input
            questionSeq.extend(batch.encoderSeqs)
        # Run the model
        ops, feedDict = self.model.step(batch)
        output = sess.run(ops[0], feedDict)  # TODO: Summarize the output too (histogram, ...)
        print("output")
        #print(output)
        answer = self.textData.deco2sentence(output)
        print("multi model answer: " + str(answer))
        return answer

    def daemonPredictMultiModel(self, sentence, sess):
        return self.textData.sequence2str(
            self.singlePredictMultiModel(sentence, sess),
            clean=True
    )

    def managePreviousModel(self, sess, modelTag):
        #modelName = self._getModelName()
        modelName = self.modelDir + '-' + modelTag + "/model" + self.MODEL_EXT
        if os.listdir("save"):
            self.saver.restore(sess, modelName)  # Will crash when --reset is not activated and the model has not been saved yet
        else:
            print('No previous model found, starting from clean directory: {}'.format(self.modelDir))

    def main(self, args=None):
        ## （1）参数及对象初始化
        # General initialisation
        argsUtil = ArgsUtil()
        argsUtil.parseArgs(argsUtil, args)
        self.args = argsUtil.args
        #self.loadModelParams()  # Update the self.modelDir and self.globStep, for now, not used when loading Model (but need to be called before _getSummaryName)

        if not self.args.rootDir:
            self.args.rootDir = os.getcwd()  # Use the current working directory

        self.textData = TextData(self.args)
        self.model = Model(self.args, self.textData)
        # Saver/summaries
        self.writer = tf.summary.FileWriter("save/model-" + self.args.modelTag)
        self.saver = tf.train.Saver(max_to_keep=200)

        # Running session
        self.sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=False)  
        )  
        self.sess2 = tf.Session(config=tf.ConfigProto(
            log_device_placement=False)  
        )  
        ## （2）初始化tf变量
        self.sess.run(tf.global_variables_initializer())
        self.sess2.run(tf.global_variables_initializer())
        ## （3）tf session预测
        self.managePreviousModel(self.sess, self.args.modelTag)
        self.managePreviousModel(self.sess2, "xiaohuang")

inference = InferenceObject()
#inference.main(args = ['--test', '--modelTag', 'xiaohuang','--vocabularySize', '1000','--corpus', 'xiaohuang',  "--numEpochs", "10", "--maxLength", "10", "--filterVocab", "0"])
inference.main(args = ['--test', '--modelTag', 'cornell','--vocabularySize', '1000','--corpus', 'xiaohuang',  "--numEpochs", "10", "--maxLength", "10", "--filterVocab", "0"])
r1 = inference.daemonPredict("你好")
r2 = inference.daemonPredictMultiModel("你好", inference.sess2)
print("model1 answer is:" + str(r1))
print("model1 answer is:" + str(r2))

