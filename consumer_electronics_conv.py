import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import time
import os 


def new_conv_layer(input, num_input_channels, filter_size, num_filters, name): 
    with tf.variable_scope(name) as scope:
        # Shape of the filter-weights for the convolution
        shape = [filter_size, filter_size, num_input_channels, num_filters]
        # Create new weights (filters) with the given shape
        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))
        # Create new biases, one for each filter
        biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))
        # TensorFlow operation for convolution
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
        # Add the biases to the results of the convolution.
        layer += biases        
        return layer, weights


def new_relu_layer(input, name):
    with tf.variable_scope(name) as scope:
        # TensorFlow operation for convolution
        layer = tf.nn.relu(input)
        return layer

def new_fc_layer(input, num_inputs, num_outputs, name):
    with tf.variable_scope(name) as scope:
        # Create new weights and biases.
        weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
        biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
        # Multiply the input and weights, and then add the bias-values.
        layer = tf.matmul(input, weights) + biases
        return layer

def new_pool_layer(input, name):
    with tf.variable_scope(name) as scope:
        # TensorFlow operation for convolution
        layer = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return layer


def parse_function(filename, label):
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    resized_image = tf.image.resize_images(image, [64, 64])
    return resized_image, label


def train_preprocess(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.clip_by_value(image, 0.0, 0.1)   
    return image, label

rootdir = './data/train/'

labels = []
labelCtr = 0
filenames = []
batch_size = 10
currLabelStr = ''

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        imgPath = os.path.join(subdir, file)
        path = os.path.dirname(imgPath)
        labelStr = os.path.basename(path)
        # print('imgPath', imgPath)
        # print('path', path)
        # print('labelStr', labelStr)
        if not currLabelStr == labelStr:
            labelCtr += 1
            currLabelStr = labelStr
        if imgPath.endswith('.jpg') or imgPath.endswith('.png') or imgPath.endswith('.jpeg'):    
            #print(file)
            #os.remove(imgPath)
            
            filenames.append(imgPath)
            labels.append(labelCtr)


num_epochs = 10
batch_size = 10

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.shuffle(len(filenames))
dataset = dataset.map(parse_function, num_parallel_calls=4)
dataset = dataset.map(train_preprocess, num_parallel_calls=4)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(1)

#Placeholder variable for the input images
x = tf.placeholder(tf.float32, shape=[None, 64*64], name='X')
print('x shape', tf.shape(x))
# Reshape it into [num_images, img_height, img_width, num_channels]
x_image = tf.reshape(x, [batch_size, 64, 64, 3])
# Placeholder variable for the true labels associated with the images
y_true = tf.placeholder(tf.int32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


# Convolutional Layer 1
layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=3, filter_size=5, num_filters=10, name ="conv1")
# Pooling Layer 1
layer_pool1 = new_pool_layer(layer_conv1, name="pool1")
# RelU layer 1
layer_relu1 = new_relu_layer(layer_pool1, name="relu1")
# Convolutional Layer 2
layer_conv2, weights_conv2 = new_conv_layer(input=layer_relu1, num_input_channels=10, filter_size=5, num_filters=20, name= "conv2")
# Pooling Layer 2
layer_pool2 = new_pool_layer(layer_conv2, name="pool2")
# RelU layer 2
layer_relu2 = new_relu_layer(layer_pool2, name="relu2")
# Flatten Layer
num_features = layer_relu2.get_shape()[1:4].num_elements()
layer_flat = tf.reshape(layer_relu2, [-1, num_features])
# Fully-Connected Layer 1
layer_fc1 = new_fc_layer(layer_flat, num_inputs=num_features, num_outputs=128, name="fc1")
# RelU layer 3
layer_relu3 = new_relu_layer(layer_fc1, name="relu3")
# Fully-Connected Layer 2
layer_fc2 = new_fc_layer(input=layer_relu3, num_inputs=128, num_outputs=10, name="fc2")
# Use Softmax function to normalize the output
with tf.variable_scope("Softmax"):
    y_pred = tf.nn.softmax(layer_fc2)
    y_pred_cls = tf.argmax(y_pred, dimension=1)
#Use Cross entropy cost function
with tf.name_scope("cross_ent"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logitsv2(logits=layer_fc2, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)

#Use Adam Optimizer
with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

#Accuracy
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the FileWriter
writer = tf.summary.FileWriter("Training_FileWriter/")
writer1 = tf.summary.FileWriter("Validation_FileWriter/")

# Add the cost and accuracy to summary
tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)
    # Loop over number of epochs
    for epoch in range(num_epochs):
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        start_time = time.time()
        train_accuracy = 0
        for batch in range(0, int(len(labels)/batch_size)):
            # Get a batch of images and labels
            x_batch, y_true_batch = sess.run(next_element)
            # Put the batch into a dict with the proper names for placeholder variables
            print('batch #', batch)
            print('y_true_batch len', len(y_true_batch))

            y_true_batch = np.reshape(y_true_batch, (-1, 10))
            
            # x_batch = np.reshape(x_batch, 4096)
            feed_dict_train = {x_image: x_batch, y_true: y_true_batch}
            # print('x type', type(x)) # tensor
            # Run the optimizer using this batch of training data.
            sess.run(optimizer, feed_dict=feed_dict_train)
            # Calculate the accuracy on the batch of training data
            train_accuracy += sess.run(accuracy, feed_dict=feed_dict_train)
            # Generate summary with the current batch of data and write to file
            summ = sess.run(merged_summary, feed_dict=feed_dict_train)
            writer.add_summary(summ, epoch*int(len(labels)/batch_size) + batch)       
        train_accuracy /= int(len(labels)/batch_size)
        
        # Generate summary and validate the model on the entire validation set
        # summ, vali_accuracy = sess.run([merged_summary, accuracy], feed_dict={x:data.validation.images, y_true:data.validation.labels})
        # writer1.add_summary(summ, epoch)
        end_time = time.time()
        print("Epoch "+str(epoch+1)+" completed : Time usage "+str(int(end_time-start_time))+" seconds")
        print("\tAccuracy:")
        print ("\t- Training Accuracy:\t{}".format(train_accuracy))
        # print ("\t- Validation Accuracy:\t{}".format(vali_accuracy))
