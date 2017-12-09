import tensorflow as tf

num_inputs = 1
num_labels = 1

inputType = tf.placeholder(shape=[None, num_inputs], dtype=tf.float32)
outputType = tf.placeholder(shape=[None, num_labels], dtype=tf.float32)

def buildNetwork(ins):
    network = tf.layers.dense(name="leon", inputs=ins, units=num_labels)
    return network


network = buildNetwork(inputType)
loss = tf.square(network - outputType)
vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="leon/bias:0") #  "leon/bias:0"
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss=loss, var_list=vars)

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

inputs = [[0], [1]]
outputs = [[0], [2]]

brainSaver = tf.train.Saver()
graphWriter = tf.summary.FileWriter("output/", sess.graph)

for i in range(500):
    print(sess.run(network, feed_dict={inputType: inputs}))
    sess.run(fetches=[train_op, loss], feed_dict={inputType: inputs, outputType: outputs})

brainSaver.save(sess, "brains/")
graphWriter.close()








