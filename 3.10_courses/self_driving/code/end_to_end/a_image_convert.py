import os
import cv2
import params
import numpy as np
import pandas as pd
import utils

def img_preprocess(img, color_mode='RGB', flip=True):
    """
    Processes the image and returns it
    :param img: The image to be processed
    :return: Returns the processed image
    """
    ## Chop off 1/3 from the top and cut bottom 150px(which contains the head of car)
    shape = img.shape
    ratio = params.img_height / params.img_width
    h1 = int(shape[0]/3)
    h2 = shape[0]-150
    w = (h2 - h1) / ratio
    padding = int(round((img.shape[1] - w) / 2))
    img = img[h1:h2, padding:-padding]
    ## Resize the image
    img = cv2.resize(img, (params.img_width, params.img_height), interpolation=cv2.INTER_AREA)
    if color_mode == 'YUV':
    	img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    if flip:
    	img = cv2.flip(img, 1)

    return img