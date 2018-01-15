import tensorflow as tf

#
num_inputs = 1
num_labels = 1

inputType = tf.placeholder(shape=[None, num_inputs], dtype=tf.float32)
outputType = tf.placeholder(shape=[None, num_labels], dtype=tf.float32)


def buildNetwork(ins):
    network = tf.layers.dense(name="vars1", inputs=ins, units=num_labels)
    return network




network = buildNetwork(inputType)
loss = tf.square(outputType - network)
vars1 = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="vars1/kernel") #  "leon/kernel:0"
vars2 = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="leon2") #  "leon/bias:0"
train_op = tf.train.AdamOptimizer(0.1).minimize(loss=loss)#, var_list=[vars1, vars2])

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

# Set the expected inputs and outputs.
inputs = [[0], [1], [3], [10], [20]] #x values
outputs = [[0], [1], [9], [100], [400]] #expected y values

inputs = [[0], [1], [2]] #x values
outputs = [[0], [1], [2]] #expected y values

# Setup the saver and graph writer.
brainSaver = tf.train.Saver(var_list={"vars1": vars1[0]})
graphWriter = tf.summary.FileWriter("output/", sess.graph)

#brainSaver.restore(sess, "brains/test.txt")

# Print the initial values of the network.
print("\n Initial Network [weight, bias]:")
print(sess.run(vars1))
print(sess.run(vars2))

# Train the network for N training events.
for i in range(100):
    print(sess.run(network, feed_dict={inputType: inputs})) #print the results of output run.
    sess.run(fetches=[train_op, loss], feed_dict={inputType: inputs, outputType: outputs})

# Print a single sample output.
print("\n Final Outputs:")
print(sess.run(network, feed_dict={inputType: inputs}))
print(sess.run(network, feed_dict={inputType: [[5]]}))

# Print results of network.
print("\n Final Network [weight, bias]:")
print(sess.run(vars1))
print(sess.run(vars2))

# Save the trained network values and write to the tensorboard.
brainSaver.save(sess, "brains/test.txt")
graphWriter.close()








