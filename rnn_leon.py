import tensorflow as tf

train_input = [[[0]], [[0]]]
train_output = [[0], [0]]
test_input = train_input
test_output = train_output
inputType = tf.placeholder(tf.float32, [None, 1, 1])  # time_steps, batch_size, num_features
outputType = tf.placeholder(tf.float32, [None, 1])


num_hidden = 2
cell = tf.nn.rnn_cell.LSTMCell(num_units=num_hidden, state_is_tuple=True)

# 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=inputType, dtype=tf.float32)
outputs = tf.transpose(outputs, [1, 0, 2])

last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
weight = tf.Variable(tf.truncated_normal([num_hidden, int(outputType.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[outputType.get_shape()[1]]))

prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
cross_entropy = -tf.reduce_sum(outputType * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
train_op = tf.train.AdamOptimizer().minimize(cross_entropy)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

val = sess.run(prediction, {inputType: [[[0]]]})
print(val)

epochs = 100
for i in range(epochs):
    sess.run(fetches=train_op, feed_dict={inputType: train_input, outputType: train_output})

val = sess.run(prediction, {inputType: [[[0]]]})
print(val)

sess.close()


