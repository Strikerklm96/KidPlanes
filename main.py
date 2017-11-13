
from __future__ import print_function
import tensorflow as tf
import numpy as np
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
    with tf.name_scope('leon'):
        dense1 = tf.layers.conv2d(inputs=in_shape, filters=6, kernel_size=(5, 5), strides=(1, 1), activation=tf.nn.sigmoid)  #5,5 window, increments window by 1 each dim, 6 target filters
        flat = tf.layers.flatten(in_shape)
        # dense1 = tf.layers.dense(inputs=in_shape, units=30, activation=tf.nn.sigmoid)
        out = tf.layers.dense(inputs=flat, units=num_labels, activation=tf.nn.sigmoid)
        tf.summary.scalar("out", out)
        return out


# tf Graph input
networkInput = tf.placeholder(shape=[100, 28, 28, 1], dtype=tf.float32)
#networkInput = tf.placeholder(shape=[100, 784], dtype=tf.float32)

networkOutput = tf.placeholder(shape=[100, num_labels], dtype=tf.float32)

# Construct model
logits = network(in_shape=networkInput)



# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=networkOutput))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
trainer = optimizer.minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("output", sess.graph)

    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples / batch_size)

        avg_cost = 0.
        j = 0
        for i in range(total_batch):
            j += 1
            images, answers = mnist.train.next_batch(batch_size)  #get next set of images and answers

            reshape = np.reshape(images, [100, 28, 28, 1])
            #reshape = np.reshape(images, [100, 784])
            summary, c = sess.run(fetches=[trainer, cost], feed_dict={networkInput: reshape, networkOutput: answers})

            avg_cost += c / total_batch
        print("Epoch: {}".format(epoch + 1) + " Cost = {:.5f}".format(avg_cost))

    print("Done Training.")

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(networkOutput, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    first = mnist.test.images[:100]
    reshape = np.reshape(first, [100, 28, 28, 1])
    print("Accuracy:", accuracy.eval({networkInput: reshape, networkOutput: mnist.test.labels[:100]}))


    writer.close()






