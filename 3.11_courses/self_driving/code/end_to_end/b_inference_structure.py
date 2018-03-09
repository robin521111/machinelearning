import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import cv2
import pylab

# Test the performance on test data
test_loss= RGB_model.evaluate(imgs_test_RGB, wheels_test_RGB, batch_size=batch_size)

model_saved_path = os.path.join(params.model_dir, "model_RGB.h5")
json_saved_path = os.path.join(params.model_dir, "model_RGB.json")
# save model weight and json
json_model = RGB_model.to_json()
with open(json_saved_path, "w") as json_file:
    json_file.write(json_model)
# save model
# save model
RGB_model.save(model_saved_path)