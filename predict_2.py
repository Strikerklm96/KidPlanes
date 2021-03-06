from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import os.path

# returns index value
def rand_pick(values, trim=0):

    if trim != 0: # trim some values off
        values = [0 if value < trim else value for value in values]

    rand = np.random.random() * np.sum(values)

    prob = 0
    for i in range(len(values)):
        prob += values[i]
        if rand <= prob:
            return i

    return len(values)-1 # if the value happens to be just higher than last prob, handle that



echo_step = -1  # by how many time steps is the input shifted to produce the output (we want to predict so we )
num_epochs = 50  # how many epochs of training should we do?
total_text = 4900000
epoch_input_length = 50000  # what is total number of input data timesteps we should generate to use per epoch?
max_start = total_text - epoch_input_length
bpl = 50  # "back prop length" how many values should be in a single training stream?
state_size = 512  # how many values should be passed to the next hidden layer
output_classes = state_size  # defines OUTPUT vector length
batch_size = 5  # how many series to process simultaneously. provides smoother training
batches_per_epoch = epoch_input_length // batch_size // bpl  # how many batches to do before starting a new epoch
# because we are out of data
learning_rate = 0.005  # how fast we try to learn (this value is important)
num_layers = 2  # how many layers of the cell type do we stack?
input_classes = state_size  # read this link
# https://stackoverflow.com/questions/47371608/cannot-stack-lstm-with-multirnncell-and-dynamic-rnn/47376568#47376568
temperature = 0.2 # outputs are divided by temperature, so 0.5 turns 2,1 into 4,2. Increasing output values linearly
# but softmax cares about the linear difference between values, so 0.5 increases confidence since (4-2) > (2-1)
# higher temperature makes the difference between values less, so it makes it more random and creative


name_list = []
name_list.extend(["layers", str(num_layers)])
name_list.extend(["state", str(state_size)])

file_name = os.path.basename(__file__)[:-3]
save_path = "./brain/" + file_name + "/"

brain_path = save_path + "model"
for i in range(len(name_list)):
    brain_path += "_" + name_list[i]
brain_path += ".ckpt"


data = ""
with open("data/shakespear.txt", "r") as myfile:
    data = myfile.read().replace('\n', '').lower()

base = 26  # how many characters
extras = 8  # how many extra characters
def charToClass(char):
    classification = np.zeros(input_classes)

    if char == ".":
        classification[base+0] = 1
    elif char == " ":
        classification[base+1] = 1
    elif char == "'":
        classification[base+2] = 1
    elif char == "\"":
        classification[base+3] = 1
    elif char == ",":
        classification[base+4] = 1
    elif char == "?":
        classification[base+5] = 1
    elif char == "!":
        classification[base+6] = 1
    elif char == ":":
        classification[base+7] = 1
    elif ord('a') <= ord(char) <= ord('z'):
        index = ord(char) - ord('a')
        classification[index] = 1
    else:
        classification[base+4] = 1

    return classification

def classToChar(arry):
    index = np.argmax(arry)
    if index == base+0:
        char = "."
    elif index == base+1:
        char = " "
    elif index == base+2:
        char = "'"
    elif index == base+3:
        char = "\""
    elif index == base+4:
        char = ","
    elif index == base+5:
        char = "?"
    elif index == base+6:
        char = "!"
    elif index == base+7:
        char = ":"
    elif 0 <= index <= 25:
        char = chr(index + ord('a'))
    else:
        char = '&'

    return char


def stringToClassList(string):
    classList = []
    for i in range(len(string)):
        classList.append(charToClass(string[i]))

    return classList


def classListToString(classList):
    string = ""
    for i in range(len(classList)):
        string += classToChar(classList[i])

    return string

dataClassList = stringToClassList(data)

def generateData():
    start = np.random.randint(0, max_start)
    inputs = np.asarray(dataClassList[start:start+epoch_input_length])
    outputs = inputs[:]  # copy it
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


predictions_series = tf.nn.softmax(tf.scalar_mul(scalar=float(1)/float(temperature), x=logits))


losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
total_loss = tf.reduce_mean(input_tensor=losses)
train_step = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(total_loss)


def TestSave(epoch, save_path):
    print("Epoch Done. Save model and write to " + save_path + str(epoch))

    saver.save(sess, brain_path)

    gen_batch_size = 1
    gen_bpl = 1
    gen_num_batches = 2000

    _current_state = np.zeros((num_layers, 2, gen_batch_size, state_size))
    batchX = np.zeros((gen_batch_size, gen_bpl, input_classes))
    batchY = np.zeros((gen_batch_size, gen_bpl, input_classes))

    batchX[0,0,1] = 1  # start the sequence

    print("Prediction:")
    chars = ""

    for i in range(gen_num_batches):

        _total_loss, _current_state, _predictions_series = sess.run(
            [total_loss, current_state, predictions_series],
            feed_dict={
                batchX_placeholder: batchX,  # input for this batch
                batchY_placeholder: batchY,  # output (answers) for this batch
                init_state: _current_state
            })

        _predictions_series = np.reshape(a=_predictions_series, newshape=[gen_batch_size, gen_bpl, output_classes])

        x = rand_pick(values=_predictions_series[0, :][0],trim=0)  # accessing 0 batch at end

        _predictions_series[0, :][0] = np.zeros(input_classes)
        _predictions_series[0, :][0][x] = 1

        chars += classListToString(_predictions_series[0, :])

        batchX[0, :] = _predictions_series[0, :]  # set the new input to be the last output

    text_file_name = "output" + str(epoch) + ".txt"
    text_file = open(save_path + text_file_name, "w")
    text_file.write(chars)
    text_file.close()
    print(chars[:150])


def decode(coded):
    vals = np.zeros(len(coded))
    for i in range(len(coded)):
        vals[i] = (np.argmax(coded[i]))

    return vals

saver = tf.train.Saver()

np.set_printoptions(precision=1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if os.path.isfile(brain_path + ".index"):
        print("Restoring Model")
        saver.restore(sess, brain_path)
    else:
        print("No Model Found")

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

            if batch_i % 10 == 0:
                print("Step:", batch_i, "Loss:", _total_loss)

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

        TestSave(epoch, save_path)





