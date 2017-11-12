from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

num_labels = 10
num_inputs = 784


# Create model
def network(in_shape):
    dense1 = tf.layers.dense(inputs=in_shape, units=30, activation=tf.nn.sigmoid)
    out = tf.layers.dense(inputs=dense1, units=num_labels, activation=tf.nn.sigmoid)
    return out


# tf Graph input
networkInput = tf.placeholder(shape=[None, num_inputs], dtype=tf.float32)
networkOutput = tf.placeholder(shape=[None, num_labels], dtype=tf.float32)

# Construct model
logits = network(in_shape=networkInput)


# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=networkOutput))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
trainer = optimizer.minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples / batch_size)

        avg_cost = 0.
        for i in range(total_batch):
            images, answers = mnist.train.next_batch(batch_size)

            _, c = sess.run(fetches=[trainer, cost], feed_dict={networkInput: images, networkOutput: answers})

            avg_cost += c / total_batch
        print("Epoch: {}".format(epoch + 1) + " Cost = {:.5f}".format(avg_cost))

    print("Done Training.")

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(networkOutput, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({networkInput: mnist.test.images, networkOutput: mnist.test.labels}))
