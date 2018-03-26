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
from pandas import Series

class TrainModel:
    
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

    def mainTrain(self, sess, args):
        mergedSummaries = tf.summary.merge_all()  # Define the summary operator (Warning: Won't appear on the tensorboard graph)
        if self.globStep == 0:  # Not restoring from previous run
            self.writer.add_graph(sess.graph)  # First time only

        # If restoring a model, restore the progression bar ? and current batch ?
        print('Start training (press Ctrl+C to save and exit)...')
        try:  # If the user exit while training, we still try to save the model
            for e in range(args.numEpochs):
                print("----- Epoch {}/{} ; (lr={}) -----".format(e+1, self.args.numEpochs, self.args.learningRate))
                batches = self.textData.getBatches()
                # TODO: Also update learning parameters eventually
                tic = datetime.datetime.now()
                for nextBatch in tqdm(batches, desc="Training"):
                    # Training pass
                    ops, feedDict = self.model.step(nextBatch)
                    assert len(ops) == 2  # training, loss
                    _, loss, summary = self.sess.run(ops + (mergedSummaries,), feedDict)
                    self.writer.add_summary(summary, self.globStep)
                    self.globStep += 1
                    # Output training status
                    if self.globStep % 10 == 0:
                        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                        tqdm.write("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (self.globStep, loss, perplexity))
                        print("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (self.globStep, loss, perplexity))
                        tqdm.write("----- Epoch {}/{} ; (lr={}) -----".format(e+1, self.args.numEpochs, self.args.learningRate))
                        print("----- Epoch {}/{} ; (lr={}) -----".format(e+1, self.args.numEpochs, self.args.learningRate))
                toc = datetime.datetime.now()
                print("Epoch finished in {}".format(toc-tic))  # Warning: Will overflow if an epoch takes more than 24 hours, and the output isn't really nicer
        except (KeyboardInterrupt, SystemExit):  # If the user press Ctrl+C while testing progress
            print('Interruption detected, exiting the program...')

        self._saveSession(self.sess)
        self.saveModelParams()

    def _saveSession(self, sess):
        tqdm.write('Checkpoint reached: saving model (don\'t stop the run)...')
        model_name = self.modelDir + '-' + self.args.modelTag + "/model" + self.MODEL_EXT
        with open(model_name, 'w') as f:  # HACK: Simulate the old model existance to avoid rewriting the file parser
            f.write('This file is used internally by DeepQA to check the model existance. Please do not remove.\n')
        self.saver.save(self.sess, model_name)  # TODO: Put a limit size (ex: 3GB for the modelDir)
        tqdm.write('Model saved.')
    
    def saveModelParams(self):
        """ Save the params of the model, like the current globStep value
        Warning: if you modify this function, make sure the changes mirror loadModelParams
        """
        config = configparser.ConfigParser()
        config['General'] = {}
        config['General']['version']  = self.CONFIG_VERSION
        config['General']['globStep']  = str(self.globStep)
        config['General']['watsonMode'] = str(self.args.watsonMode)
        config['General']['autoEncode'] = str(self.args.autoEncode)
        config['General']['corpus'] = str(self.args.corpus)

        config['Dataset'] = {}
        config['Dataset']['datasetTag'] = str(self.args.datasetTag)
        config['Dataset']['maxLength'] = str(self.args.maxLength)
        config['Dataset']['filterVocab'] = str(self.args.filterVocab)
        config['Dataset']['skipLines'] = str(self.args.skipLines)
        config['Dataset']['vocabularySize'] = str(self.args.vocabularySize)

        config['Network'] = {}
        config['Network']['hiddenSize'] = str(self.args.hiddenSize)
        config['Network']['numLayers'] = str(self.args.numLayers)
        config['Network']['softmaxSamples'] = str(self.args.softmaxSamples)
        config['Network']['embeddingSize'] = str(self.args.embeddingSize)

        # Keep track of the learning params (but without restoring them)
        config['Training (won\'t be restored)'] = {}
        config['Training (won\'t be restored)']['learningRate'] = str(self.args.learningRate)
        config['Training (won\'t be restored)']['batchSize'] = str(self.args.batchSize)
        config['Training (won\'t be restored)']['dropout'] = str(self.args.dropout)

        with open(os.path.join(self.modelDir + '-' + self.args.modelTag, self.CONFIG_FILENAME), 'w') as configFile:
            config.write(configFile)

    def managePreviousModel(self, sess):
        #modelName = self._getModelName()
        modelName = self.modelDir + '-' + self.args.modelTag + "/model" + self.MODEL_EXT
        if os.listdir("save"):
            if os.path.exists(modelName):  # Restore the model
                self.saver.restore(sess, modelName)  # Will crash when --reset is not activated and the model has not been saved yet
        else:
            print('No previous model found, starting from clean directory: {}'.format(self.modelDir))

    def main(self, args=None):
        
        ## （1）参数及对象初始化
        # General initialisation
        argsUtil = ArgsUtil()
        argsUtil.parseArgs(argsUtil, args)
        self.args = argsUtil.args

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
        ## （2）初始化tf变量
        self.sess.run(tf.global_variables_initializer())
        ## （3）tf session训练
        #self.managePreviousModel(self.sess)
        self.mainTrain(self.sess, self.args)

trainObject = TrainModel()
#trainObject.main(args = ['--modelTag', 'xiaohuang','--vocabularySize', '1000','--corpus', 'xiaohuang',  "--numEpochs", "3", "--maxLength", "10", "--filterVocab", 0])
trainObject.main(args = ['--modelTag', 'cornell','--vocabularySize', '1000','--corpus', 'cornell',  "--numEpochs", "30", "--maxLength", "10", "--filterVocab", 0])
