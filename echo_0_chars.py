# https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_classes = 26

def charToClass(char):
    classification = np.zeros(26)
    index = ord(char) - ord('a')
    classification[index] = 1
    return classification

def classToChar(arry):
    return chr(np.argmax(arry) + ord('a'))

def stringToBits(str, requestedChars):
    bits = []
    numChars = len(str)
    if(numChars >= requestedChars):
        for i in range(requestedChars):
            bits.extend(charToClass(str[i]))
        return bits
    else:
        return -1

def bitsToString(bits, maxChars):
    chars = ""
    numBits = len(bits)
    numChars = int(numBits/num_classes)
    if numChars > maxChars:
        numChars = maxChars

    for i in range(numChars):
        start = i*num_classes
        end = (i+1)*num_classes
        if end <= numBits:
            chars += classToChar(bits[start:end])
    return chars

bitArray = stringToBits("hiasdf", 2)
print(bitArray)
str = bitsToString(bitArray, 2)
print (str)

batch_size = 5

def generateData():
    text = "hi"
    end = "hi"
    num_chars = len(text)
    cause1 = stringToBits("hi", 2)#text, num_chars)
    result1 = stringToBits("hi", 2)#text[1:] + end, num_chars)
   # cause2 = [1, 1, 1]
    #result2 = [1, 1, 1]

    batchXx = []
    batchYy = []
    for i in range(1000):
        batchXx = np.append(batchXx, cause1)
        batchYy = np.append(batchYy, result1)
      #  batchXx = np.append(batchXx, cause2)
     #   batchYy = np.append(batchYy, result2)

    x = []
    y = []

    for i in range(batch_size):
        x.append(batchXx)
        y.append(batchYy)

    x = np.asarray(x)
    y = np.asarray(y)

    return x, y, num_chars

_,_, num_chars = generateData()

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = num_classes * num_chars
state_size = 6
num_batches = total_series_length // batch_size // truncated_backprop_length


batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

init_state = tf.placeholder(tf.float32, [batch_size, state_size])

W = tf.Variable(np.random.rand(state_size + 1, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros((1, state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32)
b2 = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float32)

# Unpack columns
inputs_series = tf.unstack(batchX_placeholder, axis=1)
labels_series = tf.unstack(batchY_placeholder, axis=1)

# Forward pass
current_state = init_state
states_series = []
for current_input in inputs_series:
    current_input = tf.reshape(current_input, [batch_size, 1])
    input_and_state_concatenated = tf.concat([current_input, current_state], 1)  # Increasing number of columns

    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
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
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)

print ("Starting To Train")
np.set_printoptions(precision=1)
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)

    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(num_epochs):
        x, y, _ = generateData()
        _current_state = np.zeros((batch_size, state_size))

        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
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

            one_hot_output_series = []
            single_output_series = []
            for batch_series_idx in range(5):
                one_hot_output_series = np.array(_predictions_series)[:, batch_series_idx, :]
                single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

            if batch_idx % 100 == 0:
                print("Input:")
                print(bitsToString(batchX[0], num_chars))
                print("Answer:")
                print(bitsToString(batchY[0], num_chars))
                print("Prediction:")
                print(bitsToString(single_output_series, num_chars))
                print("State:")
                print(_current_state)

            _current_state, _predictions_series = sess.run(
                [current_state, predictions_series],
                feed_dict={
                    batchX_placeholder: batchX,
                    init_state: _current_state
                })

            loss_list.append(_total_loss)

            if batch_idx % 100 == 0:
                print("Step", batch_idx, "Loss", _total_loss)
                plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()



