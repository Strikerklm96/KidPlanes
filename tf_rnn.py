#  copy pasted from https://gist.github.com/siemanko/b18ce332bde37e156034e5d3f60f8a23

import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers

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


def generate_batch(num_bits, batch_size):
    x = np.empty((num_bits, batch_size, 2))
    y = np.empty((num_bits, batch_size, 1))

    for i in range(batch_size):
        a, b, r = generate_example(num_bits)
        x[:, i, 0] = a
        x[:, i, 1] = b
        y[:, i, 0] = r
    return x, y

# real program begins
INPUT_SIZE = 2  # 2 bits per timestep
RNN_HIDDEN_SIZE = 20
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
rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputType, initial_state=cell.zero_state(batch_size, tf.float32), time_major=True)




predicted_outputs = tf.map_fn(final_projection, rnn_outputs)
error = -(outputType * tf.log(predicted_outputs + TINY) + (1.0 - outputType) * tf.log(1.0 - predicted_outputs + TINY))
error = tf.reduce_mean(error)
train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)
accuracy = tf.reduce_mean(tf.cast(tf.abs(outputType - predicted_outputs) < 0.5, tf.float32))







init_op = tf.global_variables_initializer()
session = tf.Session()
session.run(init_op)

NUM_BITS = 10
ITERATIONS_PER_EPOCH = 100
BATCH_SIZE = 16
test_x, test_y = generate_batch(num_bits=NUM_BITS, batch_size=100)

for epoch in range(8):
    epoch_error = 0
    for _ in range(ITERATIONS_PER_EPOCH):
        x, y = generate_batch(num_bits=NUM_BITS, batch_size=BATCH_SIZE)
        result = session.run([error, train_op], {inputType: x, outputType: y})
        epoch_error += result[0]

    epoch_error /= ITERATIONS_PER_EPOCH
    valid_accuracy = session.run(accuracy, {inputType: test_x, outputType: test_y})

    print("Epoch %d, train error: %.2f, valid accuracy: %.1f %%" % (epoch, epoch_error, valid_accuracy * 100.0))


# print(session.run(fetches=))
