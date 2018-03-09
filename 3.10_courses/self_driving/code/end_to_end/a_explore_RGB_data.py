import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import cv2
import pylab

# 数据探索
def exploreData(imgs_train_RGB, wheels_train_RGB, imgs_test_RGB, wheels_test_RGB):
    print(imgs_train_RGB.shape, wheels_train_RGB.shape)
    print(imgs_test_RGB.shape, wheels_test_RGB.shape)

    # visualize the processed picture RGB
    plt.imshow(imgs_train_RGB[0])
    pylab.show()
