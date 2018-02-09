import tensorflow as tf

train_input = [[[0], [0], [0]], [[1], [1], [1]]]
train_output = [[0], [1]]
test_input = train_input
test_output = train_output

batch_size = 2

n_input = 3
num_hidden = 5
n_output = 1

weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, n_output]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_output]))
}

inputType = tf.placeholder(tf.float32, [batch_size, n_input, n_output])  # batch_size, max_time, num_features
outputType = tf.placeholder(tf.float32, [None, 1])


def RNN(ins, weights, biases):
    # reshape to [1, n_input]
    ins = tf.reshape(ins, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    ins = tf.split(ins, n_input, 1)

    # 1-layer LSTM with n_hidden units.
    rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden)

    # generate prediction
    outputs, states = tf.nn.static_rnn(rnn_cell, ins, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


# outputs:[batchSize, num chained cells, outputSize]
# state:[batchSize, cell.state_size]

network = RNN(inputType, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=outputType))
train_op = tf.train.RMSPropOptimizer(learning_rate=0.02).minimize(cost)

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

for i in range(100):
    sess.run(fetches=train_op, feed_dict={inputType: train_input, outputType: train_output})
    val, other = sess.run(network, {inputType: train_input})
    print(val)
    print(other)

sess.close()
