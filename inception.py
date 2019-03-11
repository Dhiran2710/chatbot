import os, sys, time
import keras
from keras.layers.core import Layer
import keras.backend as K
import tensorflow as tf
from keras.datasets import cifar10

from keras.models import Model
from keras.layers import Conv2D, MaxPool2D,  \
    Dropout, Dense, Input, concatenate,      \
    GlobalAveragePooling2D, AveragePooling2D,\
    Flatten

import cv2 
import numpy as np 
from keras.datasets import cifar10 
from keras import backend as K 
from keras.utils import np_utils

import math 
from keras.optimizers import SGD 
from keras.callbacks import LearningRateScheduler

num_epochs = 10
batch_size = 10

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
    return filenames, labels, size

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
    files, labels, size =  load_filenames_and_labels(root_dir)
    print('# of files', len(files))
    print('loading data from', root_dir)
    # print('filenames', files)
    # print('labels', labels)
    # Create training dataset
    dataset = tf.data.Dataset.from_tensor_slices((files, labels))
    dataset = dataset.shuffle(len(files))
    dataset = dataset.map(parse_function, num_parallel_calls=4)
    dataset = dataset.map(preprocess_image, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    print(dataset.output_types)
    print(dataset.output_shapes)
    print('dataset from', root_dir, 'loaded')
    return dataset, size

train_dir = './data/train/'
test_dir = './data/test/'

train_dataset, train_dataset_size = load_data(train_dir)
test_dataset, test_dataset_size = load_data(test_dir)

# Inception NN Model modules 

def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    
    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    
    return output
	
 
# Create the GoogleNet 

kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0.2)

input_layer = Input(shape=(224, 224, 3))

x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=64,
                     filters_3x3_reduce=96,
                     filters_3x3=128,
                     filters_5x5_reduce=16,
                     filters_5x5=32,
                     filters_pool_proj=32,
                     name='inception_3a')

x = inception_module(x,
                     filters_1x1=128,
                     filters_3x3_reduce=128,
                     filters_3x3=192,
                     filters_5x5_reduce=32,
                     filters_5x5=96,
                     filters_pool_proj=64,
                     name='inception_3b')

x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=192,
                     filters_3x3_reduce=96,
                     filters_3x3=208,
                     filters_5x5_reduce=16,
                     filters_5x5=48,
                     filters_pool_proj=64,
                     name='inception_4a')


x1 = AveragePooling2D((5, 5), strides=3)(x)
x1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
x1 = Flatten()(x1)
x1 = Dense(1024, activation='relu')(x1)
x1 = Dropout(0.7)(x1)
x1 = Dense(10, activation='softmax', name='auxilliary_output_1')(x1)

x = inception_module(x,
                     filters_1x1=160,
                     filters_3x3_reduce=112,
                     filters_3x3=224,
                     filters_5x5_reduce=24,
                     filters_5x5=64,
                     filters_pool_proj=64,
                     name='inception_4b')

x = inception_module(x,
                     filters_1x1=128,
                     filters_3x3_reduce=128,
                     filters_3x3=256,
                     filters_5x5_reduce=24,
                     filters_5x5=64,
                     filters_pool_proj=64,
                     name='inception_4c')

x = inception_module(x,
                     filters_1x1=112,
                     filters_3x3_reduce=144,
                     filters_3x3=288,
                     filters_5x5_reduce=32,
                     filters_5x5=64,
                     filters_pool_proj=64,
                     name='inception_4d')


x2 = AveragePooling2D((5, 5), strides=3)(x)
x2 = Conv2D(128, (1, 1), padding='same', activation='relu')(x2)
x2 = Flatten()(x2)
x2 = Dense(1024, activation='relu')(x2)
x2 = Dropout(0.7)(x2)
x2 = Dense(10, activation='softmax', name='auxilliary_output_2')(x2)

x = inception_module(x,
                     filters_1x1=256,
                     filters_3x3_reduce=160,
                     filters_3x3=320,
                     filters_5x5_reduce=32,
                     filters_5x5=128,
                     filters_pool_proj=128,
                     name='inception_4e')

x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=256,
                     filters_3x3_reduce=160,
                     filters_3x3=320,
                     filters_5x5_reduce=32,
                     filters_5x5=128,
                     filters_pool_proj=128,
                     name='inception_5a')

x = inception_module(x,
                     filters_1x1=384,
                     filters_3x3_reduce=192,
                     filters_3x3=384,
                     filters_5x5_reduce=48,
                     filters_5x5=128,
                     filters_pool_proj=128,
                     name='inception_5b')

x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)

x = Dropout(0.4)(x)

x = Dense(10, activation='softmax', name='output')(x)

model = Model(input_layer, [x, x1, x2], name='inception_v1')

# model.summary()


# Loss function for each output layer
# Weight assigned to that output layer
# Optimization function with weight decay after every 8 epochs
# Evaluation metric

epochs = 25
initial_lrate = 0.01

def decay(epoch, steps=100):
    initial_lrate = 0.01
    drop = 0.96
    epochs_drop = 8
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

sgd = SGD(lr=initial_lrate, momentum=0.9, nesterov=False)

lr_sc = LearningRateScheduler(decay, verbose=1)



#Placeholder variable for the input images
x = tf.placeholder(tf.float32, shape=[None, 64*64], name='X')
print('x shape', tf.shape(x))
# Reshape it into [num_images, img_height, img_width, num_channels]
x_image = tf.reshape(x, [batch_size, 64, 64, 3])
# Placeholder variable for the true labels associated with the images
y_true = tf.placeholder(tf.int32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)



# model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], loss_weights=[1, 0.3, 0.3], optimizer=sgd, metrics=['accuracy'])
# Use Softmax function to normalize the output
with tf.variable_scope("Softmax"):
    y_pred = tf.nn.softmax(x)
    y_pred_cls = tf.argmax(y_pred, dimension=1)
# Cross entropy cost function 
with tf.variable_scope('cross_ent'):
    cross_ent = tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y_true)
    cross_ent = tf.reduce_mean(cross_ent)
    
# Optimizer     
with tf.variable_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_ent)

# Accuracy
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('TRAINABLE VARIALBLES', tf.trainable_variables())
sys.exit()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epochs):
        iterator = train_dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        start_time = time.time()
        train_accuracy = 0
        num_batches = int(train_dataset_size/batch_size)
        for batch in range(num_batches):
            x_batch, y_true_batch = sess.run(next_element)
            print('batch #', batch)
            print('y_true_batch len', len(y_true_batch))
            y_true_batch = np.reshape(y_true_batch, (-1, 10))
            feed_dict_train = {x_image: x_batch, y_true: y_true_batch}
            # Run optimizer using this batch of training data
            _, c = sess.run([optimizer, cross_ent], feed_dict=feed_dict_train)
            # Compute average loss
            avg_cost += c / num_batches
            # Calculate the accuracy on the batch of training data
            train_accurary += sess.run(accuracy, feed_dict=feed_dict_train)

            # summary = sess.run(merged_summary, feed_dict=feed_dict_train)
        train_accuracy /= int(len(labels)/batch_size)

        end_time = time.time()
        print("Epoch "+str(epoch+1)+" completed : Time usage "+str(int(end_time-start_time))+" seconds")
        print("\tAccuracy:")
        print ("\t- Training Accuracy:\t{}".format(train_accuracy))
