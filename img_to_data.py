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

def clean_files_and_labels(files, labels):
    num_correct_shape = 0
    num_incorrect_shape = 0
    correct_files = []
    correct_labels = []
    for i in range(len(files)):
        data = mpimg.imread(files[i])
        if(data.shape == (100, 100, 3)):
            correct_files.append(files[i])
            correct_labels.append(labels[i])
            num_correct_shape += 1
        else: 
            num_incorrect_shape += 1
    print('WARNING:', num_incorrect_shape, 'IMAGES NOT INCLUDED DUE TO INCORRECT SHAPE')
    return correct_files, correct_labels, num_correct_shape, num_incorrect_shape

def load_data(root_dir):
    img_files, img_labels, num_img_labels =  load_filenames_and_labels(root_dir)
    print('# of files before cleaning', len(img_files))
    # Caution: Assuming that the number of labels remains the same after removing invalid files and corresponding labels!
    img_files, img_labels, num_correct_shape, num_incorrect_shape = clean_files_and_labels(img_files, img_labels)
    num_img_files = len(img_files)
    print('# of files after cleaning', num_img_files)
    print('loading image files from', root_dir)
    
    img_data = np.zeros(shape=(num_img_files, 100, 100, 3))
    for i in range(num_img_files):
        img_data[i] = mpimg.imread(img_files[i])

    img_labels = np_utils.to_categorical(img_labels, num_img_labels)
    return img_data, img_labels
