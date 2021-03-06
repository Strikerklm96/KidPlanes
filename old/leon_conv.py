


""" Convolutional Neural Network.
Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
This example is using TensorFlow layers API, see 'convolutional_network_raw'
example for a raw implementation with variables.
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

import tensorflow as tf

# Training Parameters
learning_rate = 0.001
num_steps = 2000  # how many batches to do (number of backprop operations)
batch_size = 128

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units


# Create the neural network
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['images']

        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(inputs=x, filters=1, kernel_size=(5, 5), strides=(1, 1), activation=tf.nn.sigmoid)
       # conv1 = tf.layers.conv2d(inputs=conv1, filters=6, kernel_size=(5, 5), strides=(1, 1), activation=tf.nn.sigmoid)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv1)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)

    return out


def addl(logits):
    out = tf.layers.dense(logits, num_classes)
    return out


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train1 = conv_net(features, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_addl = addl(logits_train1)


    logits_test1 = conv_net(features, num_classes, dropout, reuse=True,
                           is_training=False)
    logits_addl3 = addl(logits_test1)

    # Predictions
    pred_classes = tf.argmax(logits_addl3, axis=1)
    pred_probas = tf.nn.softmax(logits_addl3)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_addl, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

# Build the Estimator
run_config = tf.contrib.learn.RunConfig()
# run_config = run_config.replace(model_dir="brains")
estimatorModel = tf.estimator.Estimator(model_fn=model_fn, config=run_config)

# s = estimatorModel.get_variable_names()
# estimatorModel.get_variable_value()
# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images}, y=mnist.train.labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)
# Train the Model
estimatorModel.train(input_fn, steps=num_steps)


# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images}, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
e = estimatorModel.evaluate(input_fn)



print("Testing Accuracy:", e['accuracy'])





