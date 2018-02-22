from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



echo_step = -1  # by how many time steps is the input shifted to produce the output (we want to predict so we )
num_epochs = 5  # how many epochs of training should we do?
epoch_input_length = 50000  # what is total number of input data timesteps we should generate to use per epoch?
bpl = 30  # "back prop length" how many values should be in a single training stream?
state_size = 50  # how many values should be passed to the next hidden layer
output_classes = state_size  # defines OUTPUT vector length
batch_size = 5  # how many series to process simultaneously. provides smoother training
batches_per_epoch = epoch_input_length // batch_size // bpl  # how many batches to do before starting a new epoch
# because we are out of data
learning_rate = 0.1  # how fast we try to learn (this value is important)
num_layers = 2  # how many layers of the cell type do we stack?
input_classes = state_size  # read this link
# https://stackoverflow.com/questions/47371608/cannot-stack-lstm-with-multirnncell-and-dynamic-rnn/47376568#47376568
output_classes_real = 8  # lets us trim off the classes that aren't used from the data generation and display

batchX = np.zeros((1, 1, input_classes))
batchY = np.zeros((1, 1, input_classes))

batchX[0,0,1] = 1  # start the sequence

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
        v = generateClassVector(i % output_classes_real)
        inputs[i] = v


    outputs = []
    for i in range(len(inputs)):
        outputs.append(inputs[i])

    outputs = np.roll(a=outputs, shift=echo_step, axis=0)  # just shifts the whole bit list over by echo_step

    # reshape this into a 2d vector where each entry has batch_size elements and an unknown (-1) number of entries in it
    inputs = inputs.reshape((batch_size, -1, input_classes))
    outputs = outputs.reshape((batch_size, -1, output_classes))

    return inputs, outputs  # have shapes[batch_size, (remainder)] in this case, [5, 10000]


# input, output, and state types [batch_size, bpl, input_classes]
# keeping the values as None lets it be dynamic so we can do 1 char at a time later!
batchX_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, None, input_classes])
batchY_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, None, output_classes])

# tuple size is 2
init_state = tf.placeholder(tf.float32, [num_layers, 2, None, state_size])
layers = tf.unstack(init_state, axis=0)
rnn_tuple_state = tuple(
         [tf.nn.rnn_cell.LSTMStateTuple(layers[idx][0], layers[idx][1])
          for idx in range(num_layers)]
)

# these are necessary, otherwise you have no way of converting state to
output_weight = tf.Variable(np.random.rand(state_size, output_classes), dtype=tf.float32)
output_bias = tf.Variable(np.zeros(shape=(1, output_classes)), dtype=tf.float32)


cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
states_series, current_state = tf.nn.dynamic_rnn(cell=cell, inputs=batchX_placeholder, initial_state=rnn_tuple_state, time_major=False)

states_series = tf.reshape(states_series, [-1, state_size])
logits = tf.matmul(states_series, output_weight) + output_bias  # logits = 150, 50
labels = batchY_placeholder


predictions_series = tf.nn.softmax(logits) # we don't need to do fancy prediction recording, just remember the values


losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
total_loss = tf.reduce_mean(input_tensor=losses)
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
    sess.run(tf.global_variables_initializer())

    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

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


            if batch_i % 100 == 0:
                print("Step:", batch_i, "Loss:", _total_loss)
                # update the plots
                #plot(loss_list, _predictions_series, batchX, batchY)

                if batch_i % 400 == 0:
                    mini_batch_prediction = []
                    rounded_prediction = []
                    batch_series_i = 2

                    _predictions_series = np.reshape(a=_predictions_series, newshape=[batch_size, bpl, output_classes])
                    rounded_input = decode(batchX[batch_series_i, :])
                    rounded_answer = decode(batchY[batch_series_i, :])
                    rounded_prediction = decode(_predictions_series[batch_series_i, :])

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

    print("\n\nStarting New thing\n\n")


    gen_batch_size = 1
    gen_bpl = 1
    gen_num_batches = 20

    _current_state = np.zeros((num_layers, 2, gen_batch_size, state_size))
    batchX = np.zeros((gen_batch_size, gen_bpl, input_classes))
    batchY = np.zeros((gen_batch_size, gen_bpl, input_classes))

    batchX[0,0,1] = 1  # start the sequence

    for i in range(gen_num_batches):

        _total_loss, _current_state, _predictions_series = sess.run(
            [total_loss, current_state, predictions_series],
            feed_dict={
                batchX_placeholder: batchX,  # input for this batch
                batchY_placeholder: batchY,  # output (answers) for this batch
                init_state: _current_state
            })

        _predictions_series = np.reshape(a=_predictions_series, newshape=[gen_batch_size, gen_bpl, output_classes])
        rounded_input = decode(batchX[0, :])
        rounded_answer = decode(batchY[0, :])
        rounded_prediction = decode(_predictions_series[0, :])

        batchX[0, :] = _predictions_series[0, :]  # set the new input to be the last output

        # The first few are bad since the state value is nonsense.
        print("Prediction:")
        print("[", end="")
        print(*rounded_prediction, sep=" ", end="")
        print("]")


plt.ioff()
plt.show()
