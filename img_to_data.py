import os
import tensorflow as tf

from keras.layers import Conv2D, MaxPool2D,  \
    Dropout, Dense, Input, concatenate,      \
    GlobalAveragePooling2D, AveragePooling2D,\
    Flatten

import numpy as np
import matplotlib.image as mpimg
from keras.utils import np_utils



def load_filenames_and_labels(rootdir):
    size = 0
    filenames = []
    labels = []
    labelCtr = 0
    currLabelStr = ''
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            imgPath = os.path.join(subdir, file)
            path = os.path.dirname(imgPath)
            labelStr = os.path.basename(path)
            if not currLabelStr == labelStr:
                labelCtr += 1
                currLabelStr = labelStr 
            filenames.append(imgPath)
            labels.append(labelCtr)
            size += 1
    num_labels = labelCtr + 1
    return filenames, labels, num_labels

def preprocess_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.clip_by_value(image, 0.0, 0.1)   
    return image, label

def parse_function(filename, label):   
    print('FILENAME', filename)
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    resized_image = tf.image.resize_images(image, [64, 64])
    return resized_image, label
    
def load_data(root_dir):
    files, labels, num_labels =  load_filenames_and_labels(root_dir)
    print('# of files', len(files))
    print('loading data from', root_dir)
    img_data = np.zeros(shape=(len(files), 100, 100, 3))
    num_wrong_shape = 0
    for i in range(len(files)):
        data = mpimg.imread(files[i])
        if(data.shape == (100, 100, 3)):
            img_data[i] = mpimg.imread(files[i])
        else:
            # print(files[i]) <--- for printing the images that do not have shape (100, 100, 3) which we do not include in the training/testing data. 
            num_wrong_shape += 1
    print('WARNING:', num_wrong_shape, 'IMAGES NOT INCLUDED DUE TO INCORRECT SHAPE')
    labels = np_utils.to_categorical(labels, num_labels)
    return img_data, labels
