"""
    The problem we are trying to solve is adding two binary numbers. The
    numbers are reversed, so that the state of RNN can add the numbers
    perfectly provided it can learn to store carry in the state. Timestep t
    corresponds to bit len(number) - t.

    copy pasted from https://gist.github.com/siemanko/b18ce332bde37e156034e5d3f60f8a23
"""

import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers

def charToClass(char):
    classification = np.zeros(26)
    index = ord(char) - ord('a')
    classification[index] = 1
    return classification

def classToChar(arry):
    return chr(np.argmax(arry) + ord('a'))

# these are just for easy building of test data
def as_bytes(num, final_size):
    res = []
    for _ in range(final_size):
        res.append(num % 2)
        num //= 2
    return res


def generate_example(num_bits):
    a = random.randint(0, 2 ** (num_bits - 1) - 1)
    b = random.randint(0, 2 ** (num_bits - 1) - 1)
    res = a + b
    return (as_bytes(a, num_bits),
            as_bytes(b, num_bits),
            as_bytes(res, num_bits))


def generate_batch(num_bits, batch_num):
    ins = np.empty((num_bits, batch_num, 2))
    outs = np.empty((num_bits, batch_num, 1))

    for i in range(batch_num):
        a, b, c = generate_example(num_bits)
        ins[:, i, 0] = a
        ins[:, i, 1] = b
        outs[:, i, 0] = c
    return ins, outs


# real program begins
INPUT_SIZE = 2  # 2 bits per timestep
RNN_HIDDEN_SIZE = 3
OUTPUT_SIZE = 1  # 1 bit per timestep
TINY = 1e-6  # to avoid NaNs in logs
LEARNING_RATE = 0.01
USE_LSTM = True

inputType = tf.placeholder(tf.float32, (None, None, INPUT_SIZE))  # (time, batch, in)
outputType = tf.placeholder(tf.float32, (None, None, OUTPUT_SIZE))  # (time, batch, out)


def final_projection(inputs):
    return layers.linear(inputs, num_outputs=OUTPUT_SIZE, activation_fn=tf.nn.sigmoid)


def build_network():
    network = tf.nn.rnn_cell.BasicLSTMCell(num_units=RNN_HIDDEN_SIZE, state_is_tuple=True)
    return network


cell = build_network()
batch_size = tf.shape(inputType)[1]
rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell,
                                            inputType,
                                            initial_state=cell.zero_state(batch_size, tf.float32),
                                            time_major=True)

predicted_outputs = tf.map_fn(final_projection, rnn_outputs)
error = -(outputType * tf.log(predicted_outputs + TINY) + (1.0 - outputType) * tf.log(1.0 - predicted_outputs + TINY))
error = tf.reduce_mean(error)
train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)
accuracy = tf.reduce_mean(tf.cast(tf.abs(outputType - predicted_outputs) < 0.5, tf.float32))

init_op = tf.global_variables_initializer()
session = tf.Session()
session.run(init_op)

bits = 10
iter_per_epoch = 100
test_x, test_y = generate_batch(num_bits=bits, batch_num=100)

example_x, example_y = generate_batch(num_bits=2, batch_num=1)

print("\nInput:  ", example_x)
print("\nOutput: ", example_y)
print(session.run(fetches=[error, train_op], feed_dict={inputType: example_x, outputType: example_y}))

for epoch in range(8):
    epoch_error = 0
    for _ in range(iter_per_epoch):
        x, y = generate_batch(num_bits=bits, batch_num=16)
        result, _train_step, _current_state, _predictions_series = session.run(
            fetches=[error, train_op, rnn_states, predicted_outputs], feed_dict={inputType: x, outputType: y})
        epoch_error += result

        print("Answer:")
        print(y[0])
        print("Prediction:")
        print(_predictions_series[0])
        print("State:")
        print(_current_state)


    epoch_error /= iter_per_epoch
    valid_accuracy = session.run(fetches=accuracy, feed_dict={inputType: test_x, outputType: test_y})

    print("Epoch %d, error: %.2f, accuracy: %.f%%" % (epoch, epoch_error, valid_accuracy * 100.0))


# print(session.run(fetches=))
