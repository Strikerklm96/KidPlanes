
from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.04
training_epochs = 15
batch_size = 100

num_labels = 10
num_inputs = 784



# Create model
def inputNetwork(input):
    #in_shape = tf.reshape(in_shape, shape=[-1, 28, 28, 1])
    #dense1 = tf.layers.conv2d(inputs=in_shape, filters=1, kernel_size=(2, 2), strides=(1, 1), activation=tf.nn.sigmoid)  #5,5 window, increments window by 1 each dim, 6 target filters
    #flat = tf.layers.flatten(dense1)

    out = tf.layers.dense(inputs=input, units=30, activation=tf.nn.sigmoid)
    return out

def output(input):
    out = tf.layers.dense(inputs=input, units=num_labels, activation=tf.nn.sigmoid)
    return out

# tf Graph input
networkInput = tf.placeholder(shape=[None, 784], dtype=tf.float32)
networkOutput = tf.placeholder(shape=[None, num_labels], dtype=tf.float32)

# Construct model
preNetwork = inputNetwork(networkInput)
fullNetwork = output(preNetwork)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fullNetwork, labels=networkOutput))
trainer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("output", sess.graph)

    if(False):
        saver.restore(sess,"brains/")
    else:
        for epoch in range(training_epochs):
            total_batch = int(mnist.train.num_examples / batch_size)

            avg_cost = 0.
            j = 0
            for i in range(total_batch):
                j += 1
                images, answers = mnist.train.next_batch(batch_size)  #get next set of images and answers

                #reshape = np.reshape(images, [100, 28, 28, 1])
                reshape = images#np.reshape(images, [None, 784])
                summary, c = sess.run(fetches=[trainer, cost], feed_dict={networkInput: reshape, networkOutput: answers})

                avg_cost += c / total_batch
            print("Epoch: {}".format(epoch + 1) + " Cost = {:.5f}".format(avg_cost))

        print("Done Training.")

    saver.save(sess, "brains/")

    # Test model
    pred = tf.nn.softmax(fullNetwork)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(networkOutput, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    #first = mnist.test.images[:100]
    #reshape = np.reshape(first, [100, 28, 28, 1])
    print("Accuracy:", accuracy.eval({networkInput: mnist.test.images, networkOutput: mnist.test.labels}))

    writer.close()






