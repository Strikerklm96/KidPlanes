# improved version takes onehot and outputs onehot explicitly

# echos inputs
# can use many different output_classes
# Note that this program DOES NOT implement an LSTM.
# this program computes the new state based on the new input and old state.
# computes the output based only on the old state

# TODO why does input_classes need to equal state_size

# has more difficulty if you:
# increase output_classes
# increase bpl
# mess with learning_rate


# comments are referencing this:
# https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
# LSTM references http://colah.github.io/posts/2030-08-Understanding-LSTMs/

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

echo_step = -1  # by how many bits is the input shifted to produce the output
num_epochs = 5  # how many epochs of training should we do?
epoch_input_length = 50000  # what is total number of inputs we should generate to use on an epoch?
bpl = 30  # "back prop length" how many values should be in a single training stream?
state_size = 50  # how many values should be passed to the next hidden layer
output_classes = 8  # defines OUTPUT vector length
batch_size = 5  # how many series to process simultaneously. look at "Schematic of the training data"
# how many batches will be done to go over all the data, note that since we are using integer division: //
# not all the data will get used
batches_per_epoch = epoch_input_length // batch_size // bpl  # results in 333
learning_rate = 0.1  # rate passed to optimizer (this value is important)
num_layers = 2
input_classes = state_size # please read the link and the description
# although input_classes needs to equal state_size, output_classes doesn't, so we build the inputs as
# a set of

# https://stackoverflow.com/questions/47371608/cannot-stack-lstm-with-multirnncell-and-dynamic-rnn/47376568#47376568


def generateRandomClassVector():
    vector = np.zeros(input_classes)
    vector[np.random.randint(0, output_classes)] = 1
    return vector

def generateClassVector(i):
    vector = np.zeros(input_classes)
    vector[i] = 1
    return vector


def generateData():
    inputs = np.empty((epoch_input_length, input_classes))
    for i in range(epoch_input_length):
        v = generateClassVector(i % output_classes)
        inputs[i] = v


    outputs = []
    for i in range(len(inputs)):
        outputs.append(np.resize(inputs[i], output_classes))

    outputs = np.roll(a=outputs, shift=echo_step, axis=0)  # just shifts the whole bit list over by echo_step

    # reshape this into a 2d vector where each entry has batch_size elements and an unknown (-1) number of entries in it
    inputs = inputs.reshape((batch_size, -1, input_classes))
    outputs = outputs.reshape((batch_size, -1, output_classes))

    return inputs, outputs  # have shapes[batch_size, (remainder)] in this case, [5, 10000]


# input, output, and state types
batchX_placeholder = tf.placeholder(dtype=tf.float32, shape=[batch_size, bpl, input_classes])
batchY_placeholder = tf.placeholder(dtype=tf.int32, shape=[batch_size, bpl, output_classes])

# tuple size is 2
init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size])
layers = tf.unstack(init_state, axis=0)
rnn_tuple_state = tuple(
         [tf.nn.rnn_cell.LSTMStateTuple(layers[idx][0], layers[idx][1])
          for idx in range(num_layers)]
)

# used to compute the output given the state
output_weight = tf.Variable(np.random.rand(state_size, output_classes), dtype=tf.float32)
output_bias = tf.Variable(np.zeros(shape=(1, output_classes)), dtype=tf.float32)

# Unpack columns
# defines a basic RNN cell with a given state size
# time_major=False just dictates the order of the shape of inputs
# states_series is a tensor of shape [batch_size, bpl, state_size]
# current state isn't really used here
cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
states_series, current_state = tf.nn.dynamic_rnn(cell=cell, inputs=batchX_placeholder, initial_state=rnn_tuple_state, time_major=False)

# shape [bpl, batch_size, output_classes]
# the input has been wrapped into the states_series list
# LSTM: the states_series is storing the top line having already been through tanh, multiplying x and adding bias +
# LSTM: logits_series is the LSTM output ht, computed by tanh(state) * w2 + b
logits_series = []  # = tf.placeholder(dtype=tf.float32, shape=[bpl, batch_size, output_classes])
states_series = tf.transpose(states_series, [1, 0, 2]) # reshape the states_series because tf is Jank A.F.
for i in range(bpl):  # logits_series is basically output series with size [30, output_classes]
    logits_series.append(tf.matmul(states_series[i], output_weight) + output_bias)


# create another output, but apply (next line)
# softmax (which basically just turns the output into probabilities instead of arbitrary
# values, so they sum to 1: [0.1, 0.4] would be turned to [0.2, 0.8], or [3, 6] -> [0.33, 0.33]
# just looking at logits_series is also fine
# defines what the nn should pass back to us for this argument
predictions_series = []
for i in range(bpl):
    predictions_series.append(tf.nn.softmax(logits_series[i]))

# compute how wrong the guess is by comparing the output(logits) to the correct output (labels)
# note that the logits results in a vector that is onehot encoded, so [0 0 1 0], but labels is just the value of the
# index that should be 1, so 2. That is what sparse_softmax_cross_entropy_with_logits does
# https://stackoverflow.com/questions/37312421/tensorflow-whats-the-difference-between-sparse-softmax-cross-entropy-with-logi

labels_series = tf.unstack(batchY_placeholder, axis=1)  # reshape so it can be used in zip
losses = [tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in
          zip(logits_series, labels_series)]

# computes average value across all values in input_tensor (can do more if fed more values)
total_loss = tf.reduce_mean(input_tensor=losses)

# does backprop for us (corrects our tf variables so they are more accurate)
train_step = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(total_loss)


def decode(coded):
    vals = np.zeros(len(coded))
    for i in range(len(coded)):
        vals[i] = (np.argmax(coded[i]))

    return vals


def plot(loss_list, predictions_series, batchX, batchY):
    ax = plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)
    ax.set_ylim(ymin=0)  # always show y0 and

    for batch_series_idx in range(5):
        coded = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = decode(coded)

        plt.subplot(2, 3, batch_series_idx + 2)  # select the next plot to draw on
        plt.cla()
        plt.axis([0, bpl, 0, 2])
        left_offset = range(bpl)

        barHeight = 0.1
        nextBars = barHeight * output_classes

        print()

        plt.bar(x=left_offset, height=decode(batchX[batch_series_idx, :]) * barHeight, bottom=nextBars * 2, width=1,
                color="red")  # input

        plt.bar(x=left_offset, height=decode(batchY[batch_series_idx, :]) * barHeight, bottom=nextBars * 1, width=1,
                color="green")  # output

        plt.bar(x=left_offset, height=single_output_series * barHeight, bottom=nextBars * 0, width=1,
                color="blue")  # network guess

    plt.draw()
    plt.pause(0.0001)


np.set_printoptions(precision=1)
with tf.Session() as sess:
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    # one epoch will go through all the training data
    # each epoch will call to generate more data
    # (likely to be different data in this case since rand, but generally doesn't have to be if you have limited data)
    for epoch in range(num_epochs):
        x, y = generateData()
        # tuple size is 2
        _current_state = np.zeros((num_layers, 2, batch_size, state_size))

        print("New data, epoch:", epoch)

        sub_loss_list = []  # store the loss value because displaying every single one is silly
        for batch_i in range(batches_per_epoch):
            # find where in the data to start for this batch
            start_batch_pos = batch_i * bpl
            end_batch_pos = start_batch_pos + bpl

            # for all lists in this list, grab this range [start_batch_pos:end_batch_pos)
            batchX = x[:,
                     start_batch_pos:end_batch_pos]  # size [5, 30] because [batch_size, bpl]
            batchY = y[:,
                     start_batch_pos:end_batch_pos]  # size [5, 30] because [batch_size, bpl]



            # _total_loss is just the average loss across this batch, so a float
            # _train_step is None
            # _current_state is shape [5, 4] because [batch_size, state_size] it will be fed to next batch
            # _predictions_series has shape [30, 5, 2] because [bpl, batch_size, output_classes]
            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder: batchX,  # input for this batch
                    batchY_placeholder: batchY,  # output (answers) for this batch
                    init_state: _current_state
                })


            # keep track of the loss values so we can plot them
            sub_loss_list.append(_total_loss)
            num_loss_avg = 20  # average accross this many to prevent spikes
            if (len(sub_loss_list) >= num_loss_avg):
                averageValue = [np.average(sub_loss_list)]
                averageValue = averageValue * num_loss_avg  # repeat this value num_loss_avg times
                sub_loss_list = []
                loss_list.extend(averageValue)

            # don't display more than n in the graph
            # if (len(loss_list) > 2000):
            #    del loss_list[0]

            # every n batches, print an update
            if batch_i % 100 == 0:
                print("Step:", batch_i, "Loss:", _total_loss)
                # update the plots
                plot(loss_list, _predictions_series, batchX, batchY)

                if batch_i % 400 == 0:
                    mini_batch_prediction = []
                    rounded_prediction = []
                    batch_series_i = 2  # use the third run so the state and first few values make sense
                    # TODO why do the values still not make sense?

                    # predictions_series has shape [30, 5, 2]
                    # because [bpl, batch_size, output_classes]
                    # np.array so we can use fancy index -> [magic, python, indexing]
                    # grab all time outputs, for batch (batch_series_i) and all class output values
                    mini_batch_prediction = np.array(_predictions_series)[:, batch_series_i, :]

                    # each output is a list [output_classes]
                    # decode mini_batch_prediction outputs to go to either 0 or 1 instead of the one hot classes
                    # out[0] can be compared to 0.5 because tf.nn.softmax(logits) turned the guesses
                    # into probabilities.
                    # out[0] is the probability of 0 being the right answer
                    # out[1] is the probability of 1 being the right answer
                    rounded_input = decode(batchX[batch_series_i, :])
                    rounded_answer = decode(batchY[batch_series_i, :])
                    rounded_prediction = decode(mini_batch_prediction)

                    print("Input:")
                    print("[", end="")
                    print(*rounded_input, sep=" ", end="")
                    print("]")
                    print("Output:")
                    print("[", end="")
                    print(*rounded_answer, sep=" ", end="")
                    print("]")
                    print("Prediction:")
                    print("[", end="")
                    print(*rounded_prediction, sep=" ", end="")
                    print("]")
                    print("Resulting State:")



                   # print(_current_cell_state[batch_series_i])  # the resulting state after the run

            _current_state = np.zeros((num_layers, 2, batch_size, state_size))




print("New data, epoch:", epoch)
sub_loss_list = []  # store the loss value because displaying every single one is silly
for batch_i in range(4):
    # find where in the data to start for this batch
    start_batch_pos = batch_i * bpl
    end_batch_pos = start_batch_pos + bpl

    # for all lists in this list, grab this range [start_batch_pos:end_batch_pos)
    batchX = x[:, start_batch_pos:end_batch_pos]
    batchY = y[:, start_batch_pos:end_batch_pos]
    _total_loss, _current_state, _predictions_series = sess.run(
        [total_loss, current_state, predictions_series],
        feed_dict={
            batchX_placeholder: batchX,  # input for this batch
            batchY_placeholder: batchY,  # output (answers) for this batch
            init_state: _current_state
        })

    mini_batch_prediction = np.array(_predictions_series)[:, batch_series_i, :]
    rounded_answer = decode(batchY[batch_series_i, :])
    rounded_prediction = decode(mini_batch_prediction)

    print("Input:")
    print("[", end="")
    print(*rounded_input, sep=" ", end="")
    print("]")
    print("Output:")
    print("[", end="")
    print(*rounded_answer, sep=" ", end="")
    print("]")
    print("Prediction:")
    print("[", end="")
    print(*rounded_prediction, sep=" ", end="")
    print("]")
    print("Resulting State:")


plt.ioff()
plt.show()