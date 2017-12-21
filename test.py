import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import GNN

#
num_inputs = 784
num_labels = 10
net_width = 30

inputType = tf.placeholder(shape=[None, num_inputs], dtype=tf.float32)
outputType = tf.placeholder(shape=[None, num_labels], dtype=tf.float32)

learner0 = "layer0"
learner1 = "layer11"
learner2 = "layer2"

def buildNetwork(network):
    network = tf.layers.dense(name=learner0, inputs=network, units=net_width, activation=tf.sigmoid)
    network = tf.layers.dense(name=learner2, inputs=network, units=net_width, activation=tf.sigmoid)
    return network

def buildOutput(network):
    return tf.layers.dense(name=learner1, inputs=network, units=num_labels, activation=tf.sigmoid)


network = buildNetwork(inputType)
network = buildOutput(network)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=outputType))
train_vars0 = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=learner0) #  "leon/kernel:0"
train_vars1 = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=learner1) #  "leon/bias:0"
train_vars2 = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=learner2) #  "leon/bias:0"


#train_op0 = tf.train.AdamOptimizer(0.0).minimize(loss=loss, var_list=[train_vars0])
train_op1 = tf.train.AdamOptimizer(0.05).minimize(loss=loss, var_list=[train_vars1])
train_op2 = tf.train.AdamOptimizer(0.05).minimize(loss=loss, var_list=[train_vars2])
train_op = tf.group(train_op1, train_op2)#, train_op0)

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

# Setup the saver and graph writer.
brainSaver = tf.train.Saver(var_list={"t0_0": train_vars0[0], "t0_1": train_vars0[1]})#, "t1_0": train_vars1[0], "t1_1": train_vars1[1]})
graphWriter = tf.summary.FileWriter("output/", sess.graph)

load = True
save = True
train = True

if load:
    brainSaver.restore(sess, "brains/test.txt")

# Print the initial values of the network.
# print("\n Initial Network [weight, bias]:")
# print(sess.run(vars1))
# print(sess.run(vars2))



training_epochs = 15
batch_size = 100

with sess.as_default():
    pred = tf.nn.softmax(network)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(outputType, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # first = mnist.test.images[:100]
    # reshape = np.reshape(first, [100, 28, 28, 1])
    print("Pre-training accuracy:", accuracy.eval({inputType: mnist.test.images, outputType: mnist.test.labels}))

# Train the network for N training events.
if train:
    for x in range(1):
        #print(sess.run(vars1))

        for epoch in range(training_epochs):
            num_batches = int(mnist.train.num_examples / batch_size)

            avg_cost = 0.
            j = 0
            for i in range(num_batches):
                j += 1
                inputs, outputs = mnist.train.next_batch(batch_size)  #get next set of images and answers
                summary, c = sess.run(fetches=[train_op, loss], feed_dict={inputType: inputs, outputType: outputs})

                avg_cost += c / num_batches
            print("Epoch: {}".format(epoch + 1))
            print("Cost = {}".format(avg_cost))

        #if(x==5):
            #print("\n Final Outputs:")
            #print(sess.run(network, feed_dict={inputType: inputs}))

        #varbs = vars2 # brainSaver.save(sess, "brains/test.txt")
        #init_op = tf.global_variables_initializer()
        #vars2 = varbs# brainSaver.restore(sess, "brains/test.txt")

# Print a single sample output.
#print("\n Final Outputs:")
#print(sess.run(network, feed_dict={inputType: inputs}))

# Print results of network.
#print("\n Final Network [weight, bias]:")
#print(sess.run(vars1))
#print(sess.run(vars2))

# with sess.as_default():
#     pred = tf.nn.softmax(network)  # Apply softmax to logits
#     correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(outputType, 1))
#     accuracyCalc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#     accuracy = accuracyCalc.eval({inputType: inputs, outputType: outputs})
#     print("Accuracy:", accuracy)

with sess.as_default():
    pred = tf.nn.softmax(network)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(outputType, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # first = mnist.test.images[:100]
    # reshape = np.reshape(first, [100, 28, 28, 1])
    print("Accuracy:", accuracy.eval({inputType: mnist.test.images, outputType: mnist.test.labels}))

# Save the trained network values and write to the tensorboard.
if save:
    brainSaver.save(sess, "brains/test.txt")
graphWriter.close()








