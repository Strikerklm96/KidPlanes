# echos binary inputs

# comments are referencing this:
# https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767

# TODO name all inputs to functions

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

echo_step = 3  # by how many bits is the input shifted to produce the output
num_epochs = 100  # how many epochs of training should we do?
total_series_length = 50000  # what is total number of bits we should generate?
truncated_backprop_length = 15  # how many bits should be in a single train stream?
state_size = 4  # how many values should be passed to the next hidden layer
num_classes = 2  # TODO Having this low makes it faster and seem to split the data differently
batch_size = 5  # how many series to process simultaneously. look at "Schematic of the training data"
# how many batches will be done to go over all the data
batches_per_epoch = total_series_length // batch_size // truncated_backprop_length


def generateData():
    # 2 defines [0,1] as rand range, then how many, then rand distribution
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)  # just shifts the whole bit list over by echo_step
    y[0:echo_step] = 0  # sets the beginning values here to be 0 since they are garbage

    # reshape this into a 2d vector where each entry has batch_size elements and an unknown (-1) number of entries in it
    x = x.reshape((batch_size, -1))
    y = y.reshape((batch_size, -1))

    return x, y  # have shapes[batch_size, (remainder)] in this case, [5, 10000]


# input, output, and state types
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])
init_state = tf.placeholder(tf.float32, [batch_size, state_size])  # this is a RNN, so we need a state type too

# TODO how do these relate to the picture, this defines part of the cell, probably the powerhouse
W = tf.Variable(np.random.rand(state_size + 1, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros((1, state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32)
b2 = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float32)

# Unpack columns
# keep in mind truncated_backprop_length = 15
# this splits the [batch_size, truncated_backprop_length] tensors into (15) different tensors of shape (5,)
# these are now lists of (15) tensors, each one defining a single cell's input or output
inputs_series = tf.unstack(batchX_placeholder, axis=1)  # axis=1 says to split on the 2nd dimension (indexed on 0)
labels_series = tf.unstack(batchY_placeholder, axis=1)

# Forward pass
current_state = init_state  # current_state will be passed to each next cell
states_series = []
# define the (15) cells all joined together. Here we are basically manually unrolling the RNN
# Colors refer to "Schematic of the computations" which you should look at.
for current_input in inputs_series:

    # defines the input to this cell as having 1 input (green)
    current_input = tf.reshape(tensor=current_input, shape=[batch_size, 1])

    # just combines the input and previous state (green and blue)
    input_and_state = tf.concat([current_input, current_state], 1)

    # computes
    lhs = tf.matmul(input_and_state, W) + b #multiply, then add Broadcasted addition

    next_state = tf.tanh(lhs)
    states_series.append(next_state)
    current_state = next_state

logits_series = [tf.matmul(state, W2) + b2 for state in states_series]  # Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in
          zip(logits_series, labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.5).minimize(total_loss)


def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(5):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :] * 0.9, width=1, color="blue")  # input
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.6, width=1, color="red")  # correct output
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")  # network output

    plt.draw()
    plt.pause(0.0001)


np.set_printoptions(precision=1)
with tf.Session() as sess:
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(num_epochs):
        x, y = generateData()
        _current_state = np.zeros((batch_size, state_size))

        print("New data, epoch", epoch_idx)

        for batch_idx in range(batches_per_epoch):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = x[:, start_idx:end_idx]
            batchY = y[:, start_idx:end_idx]

            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder: batchX,
                    batchY_placeholder: batchY,
                    init_state: _current_state
                })

            loss_list.append(_total_loss)

            if batch_idx % 100 == 0:
                print("Step", batch_idx, "Loss", _total_loss)
                plot(loss_list, _predictions_series, batchX, batchY)
                if batch_idx % 500 == 0:
                    one_hot_output_series = []
                    single_output_series = []
                    for batch_series_idx in range(1):
                        one_hot_output_series = np.array(_predictions_series)[:, batch_series_idx, :]
                        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

                    print("Answer:")
                    print(batchY[0, :])
                    print("Prediction:")
                    print(single_output_series)
                    print("State:")
                    print(_current_state)

plt.ioff()
plt.show()
