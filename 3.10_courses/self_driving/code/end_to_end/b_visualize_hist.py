import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import cv2
import pylab
# set the basic parameters
import params, utils

# (2) 数据分布
# 可视化Steering Angel的分布。
# 上图表明Steering Angle大致成正态分布，小幅转角出现的次数多，大幅转角出现的次数少。

def steering_distribution():
    wheel_sig = pd.read_csv(params.data_dir + '/epoch01_steering.csv')
    wheel_sig.head()
    wheel_sig.wheel.hist(bins=50)