#!/usr/bin/env python 
import sys
import os
import time
import subprocess as sp
import itertools
## CV
import cv2
## Model
import numpy as np
import tensorflow as tf
## Tools
import utils
## Parameters
import params ## you can modify the content of params.py

import preprocess


## Test epoch
epoch_ids = [10]
## Load model
model = utils.get_model()



## Process video
for epoch_id in epoch_ids:
    print('---------- processing video for epoch {} ----------'.format(epoch_id))
    vid_path = utils.join_dir(params.data_dir, 'epoch{:0>2}_front.mp4'.format(epoch_id))
    assert os.path.isfile(vid_path)

    cap = cv2.VideoCapture(vid_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    imgs_test, wheels_test = preprocess.load_test(color_mode='RGB')
    
    machine_steering = []

    print('performing inference...')

    time_start = time.time()

    machine_steering = model.predict(imgs_test, batch_size=128, verbose=0)

    fps = frame_count / (time.time() - time_start)

    print('completed inference, total frames: {}, average fps: {} Hz'.format(frame_count, round(fps, 1)))

    print('performing visualization...')

    utils.visualize(epoch_id, machine_steering, params.out_dir,
	                        verbose=True, frame_count_limit=None)

