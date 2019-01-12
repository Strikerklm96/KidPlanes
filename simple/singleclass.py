# trains a neural network with data from the analysis folder
# you need to generate the analysis data first

from __future__ import print_function

from typing import Iterator

import numpy as np
import tensorflow as tf

from utils.FileEnumerable import FileEnumerable
from utils.Investments import Investments
from utils.TrainingData import DataPoint
from utils.TrainingData import TrainingData

learning_rate = 10
training_epochs = 4
batch_size = 20

our_invest_size = Investments.num_investment_options()
num_inputs = our_invest_size * 2
num_hidden_1 = 2
num_outputs = our_invest_size  # win loss

save_directory = "brains/{}_{}_{}_sc2bot_v2_brain.ckpt".format(num_inputs, num_hidden_1, num_outputs)


# Create model
def add_middle_layer(existing_network):
	out = tf.layers.dense(inputs=existing_network, units=num_hidden_1, activation=tf.nn.sigmoid, name="hidden")
	return out


def add_output_layer(existing_network):
	out = tf.layers.dense(inputs=existing_network, units=num_outputs, activation=tf.nn.sigmoid, name="output")
	return out


def get_training_enumerable() -> Iterator[TrainingData]:
	for _ in FileEnumerable.get_analysis_enumerable():
		yield _


def randomize_data(data_array: [DataPoint]):
	np.random.shuffle(data_array)


def all_greater_than_n(values: [int], val: int) -> bool:
	out = (values[0:our_invest_size] > val).any()
	return out


def generate_data() -> ([[int]], [[int]]):
	training_data_array: [DataPoint] = []
	for _ in FileEnumerable.get_analysis_enumerable():
		data: TrainingData = _
		for i in range(len(data.data_points) - 1):
			this_data_point: DataPoint = data.data_points[i]
			next_data_point: DataPoint = data.data_points[i + 1]
			investment_delta = np.subtract(next_data_point.inputs, this_data_point.inputs)
			if all_greater_than_n(investment_delta, 0):
				this_data_point.outputs = investment_delta[:our_invest_size]
				this_data_point.outputs = np.true_divide(this_data_point.outputs, 1000)
				this_data_point.inputs = [1, 1, 1, 1, 1, 1, 1, 1]
				this_data_point.outputs = [0, 0, 0.5, 0.25]  # np.clip(this_data_point.outputs, 0, 1)
				training_data_array = np.append(training_data_array, this_data_point)

	randomize_data(training_data_array)

	_input_array: [[int]] = []
	_output_array: [[int]] = []
	for _ in training_data_array:
		data: DataPoint = _
		_input_array.append(data.inputs)
		_output_array.append(data.outputs)

	return _input_array, _output_array


def print_manual_evaluation(session: tf.Session, network, input_type: tf.placeholder):
	test_input = [[0, 0, 600, 400, 0, 0, 600, 400]]
	test_input = [[1, 1, 1, 1, 1, 1, 1, 1]]
	# by just passing network, it implies we want the output from network
	test_output = session.run(fetches=[network], feed_dict={input_type: test_input})
	print("Odds players winning given {}: {} ".format(test_input, test_output))


def print_accuracy(session, network, input_data, output_data, input_type, output_type, before_or_after):
	# Test model
	# TODO: we need to test with data the network hasn't seen to make sure it's not over-fitting
	tests = []
	compute_correct_prediction = tf.equal(tf.argmax(network, name="argmax_output"),
	                                      tf.argmax(output_type, name="argmax_answer"), name="compare_equal")
	accuracy: tf.reduce_mean = tf.reduce_mean(tf.cast(compute_correct_prediction, "float"))
	accuracy_value = session.run(fetches=[accuracy], feed_dict={input_type: input_data, output_type: output_data})
	tests.append(accuracy_value)
	print("Accuracy {}: {:.2f}%".format(before_or_after, np.mean(tests) * 100))


def run():
	(input_data_full, output_data_full) = generate_data()

	# tf Graph input
	input_type = tf.placeholder(shape=[None, num_inputs], dtype=tf.float32)
	output_type = tf.placeholder(shape=[None, num_outputs], dtype=tf.float32)

	# Construct model
	network = add_middle_layer(input_type)
	network = add_output_layer(network)
	# Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=network, labels=output_type))
	trainer = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)

	saver = tf.train.Saver()
	with tf.Session() as session:
		session.run(tf.global_variables_initializer())

		# try to load the brain
		try:
			saver.restore(session, save_directory)
			print("Brain loaded.")  # We loaded a network!
		except ValueError:
			print("No brain found. Creating new brain.")  # There was nothing to load.

		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter("C:\\dev\\ai\\sc2\\sc2ai\\custom\sc2bot_v2\\brains\\thing", session.graph)

		print_accuracy(session, network, input_data_full, output_data_full, input_type, output_type, "before")
		print_manual_evaluation(session, network, input_type)

		print("Training.")
		for epoch in range(training_epochs):
			total_batches = int(len(input_data_full) / batch_size) - 1
			bat = 0

			# batches should actually not be all the data
			for batch in range(total_batches):
				# fetches determines what to "compute"
				# passing trainer implies you want to compute, and therefor influence, the values of the network
				c, _ = session.run(fetches=[cost, trainer],
				                   feed_dict={input_type: input_data_full[bat:bat + batch_size],
				                              output_type: output_data_full[bat:bat + batch_size]})
				bat += batch_size

			print(c)

		print("Saving.")
		saver.save(session, save_directory)

		print_accuracy(session, network, input_data_full, output_data_full, input_type, output_type, "after")
		print_manual_evaluation(session, network, input_type)


run()
