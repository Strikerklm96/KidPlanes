# trains a neural network with data from the analysis folder
# you need to generate the analysis data first

from __future__ import print_function

import numpy as np
import tensorflow as tf

learning_rate = 10
total_training_sessions = 100

num_inputs = 4
num_hidden_1 = 4
num_outputs = 4


# Create model
def add_middle_layer(existing_network):
	out = tf.layers.dense(inputs=existing_network, units=num_hidden_1, activation=tf.nn.sigmoid, name="hidden")
	return out


def add_output_layer(existing_network):
	out = tf.layers.dense(inputs=existing_network, units=num_outputs, activation=tf.nn.sigmoid, name="output")
	return out


# Data
def generate_data() -> ([[int]], [[int]]):
	_input_array = [[1, 1, 1, 1]]
	_output_array = [[1, 0, 1, 0]]

	return _input_array, _output_array


def print_manual_evaluation(session: tf.Session, network, input_type: tf.placeholder):
	test_input = [[1, 1, 1, 1]]
	# by just passing network, it implies we want the output from network
	test_output = session.run(fetches=[network], feed_dict={input_type: test_input})
	print("Output given {}: {} ".format(test_input, np.around(test_output, decimals=2)))


def print_accuracy(session, network, input_data, output_data, input_type, output_type, before_or_after):
	# Test model
	compute_error = tf.losses.mean_squared_error(predictions=network, labels=output_type)
	accuracy = tf.reduce_mean(tf.cast(compute_error, "float"))
	accuracy_value = session.run(fetches=[accuracy], feed_dict={input_type: input_data, output_type: output_data})
	print("Accuracy {}: {:.2f}%".format(before_or_after, 1.0-np.mean(accuracy_value) * 100))


def run():
	(input_data_full, output_data_full) = generate_data()

	# tf Graph input
	input_type = tf.placeholder(shape=[None, num_inputs], dtype=tf.float32)
	output_type = tf.placeholder(shape=[None, num_outputs], dtype=tf.float32)

	# Construct model
	network = add_middle_layer(input_type)
	network = add_output_layer(network)

	# Define cost and optimizer
	cost_computation = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=network, labels=output_type))
	trainer = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost_computation)

	with tf.Session() as session:
		session.run(tf.global_variables_initializer())  # randomly initialize network

		print_accuracy(session, network, input_data_full, output_data_full, input_type, output_type, "before")
		print_manual_evaluation(session, network, input_type)

		print("Training.")
		for _ in range(total_training_sessions):  # _ just means ignore this value
			# fetches determines what to "compute"
			# passing trainer implies you want to compute, and therefor influence, the values of the network
			cost, _ = session.run(fetches=[cost_computation, trainer],
			                      feed_dict={input_type: input_data_full, output_type: output_data_full})

			print(cost)

		print_accuracy(session, network, input_data_full, output_data_full, input_type, output_type, "after")
		print_manual_evaluation(session, network, input_type)


run()
