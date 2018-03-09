import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import cv2
import pylab
# set the basic parameters
import params, utils

data_dir = params.data_dir
out_dir = params.out_dir
model_dir = params.model_dir
img_dir = os.path.abspath(params.data_dir + '/images/img/')
img_height = params.img_height
img_width = params.img_width
img_channels = params.img_channels
batch_size = params.batch_size

def save_model(RGB_model):
    model_saved_path = os.path.join(params.model_dir, "model_RGB.h5")
    json_saved_path = os.path.join(params.model_dir, "model_RGB.json")
    # save model weight and json
    json_model = RGB_model.to_json()
    with open(json_saved_path, "w") as json_file:
        json_file.write(json_model)
    # save model
    RGB_model.save(model_saved_path)