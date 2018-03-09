import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import cv2
import pylab
import pickle
import params

# (1) 加载数据
# load RGB data
def loadTrainingData():
    with open (params.root + '/data1.pickle', 'rb') as f:
        data = pickle.load(f)
        imgs_train_RGB = data['imgs_train_RGB']
        wheels_train_RGB = data['wheels_train_RGB'] 
        imgs_test_RGB = data['imgs_test_RGB']
        wheels_test_RGB = data['wheels_test_RGB']
    return imgs_train_RGB, wheels_train_RGB, imgs_test_RGB, wheels_test_RGB