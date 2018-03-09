import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import cv2
import pylab

def inference(RGB_model, imgs_test_RGB, wheels_test_RGB, batch_size):
    test_loss= RGB_model.evaluate(imgs_test_RGB, wheels_test_RGB, batch_size=batch_size)
    print('Test loss is:{}'.format(test_loss))
    return test_loss