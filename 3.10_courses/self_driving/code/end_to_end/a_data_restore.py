
import params, preprocess
import utils
import pickle
import os
import a_image_convert
import a_video_capture

# load RGB train data
print("process imgs_train_RGB, wheels_train_RGB")
imgs_train_RGB, wheels_train_RGB = a_video_capture.train_data_RGB()

# load RGB test data
print("process imgs_test_RGB, wheels_test_RGB")
imgs_test_RGB, wheels_test_RGB = a_video_capture.test_data_RGB()

# load YUV train data
print("process imgs_train_YUV, wheels_train_YUV")
imgs_train_YUV, wheels_train_YUV = a_video_capture.train_data_YUV()

# load YUV test data
print("process imgs_test_YUV, wheels_test_YUV")
imgs_test_YUV, wheels_test_YUV = a_video_capture.test_data_YUV()

# set the path to save pickle file
data_root = '.'
pickle_file1 = os.path.join(data_root, 'data1.pickle')

# save the RGB data as pickle file 1
try:
    f = open(pickle_file1, 'wb')
    save1 = {
    'imgs_train_RGB': imgs_train_RGB,
    'wheels_train_RGB': wheels_train_RGB,
    'imgs_test_RGB': imgs_test_RGB,
    'wheels_test_RGB': wheels_test_RGB, 
    }
    pickle.dump(save1, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file1, ':', e)
    raise

# set the path to save pickle file
pickle_file2 = os.path.join(data_root, 'data2.pickle')

# save the YUV data as pickle file 2
try:
    f = open(pickle_file2, 'wb')
    save2 = {
    'imgs_train_YUV': imgs_train_YUV,
    'wheels_train_YUV': wheels_train_YUV, 
    'imgs_test_YUV': imgs_test_YUV,
    'wheels_test_YUV': wheels_test_YUV, 
    }
    pickle.dump(save2, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise