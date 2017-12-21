import tensorflow as tf


class GNN(object):
    sess = tf.Session()

    def __init__(self,
                 input_shape=tf.placeholder(shape=[None, 1], dtype=tf.float32),
                 output_shape=tf.placeholder(shape=[None, 1], dtype=tf.float32),
                 network_dir="brains/gnn.txt",
                 output_dir="output/"):

        self.inputShape = input_shape
        self.outputShape = output_shape
        self.network_dir = network_dir
        self.output_dir = output_dir

        self.trained_vars = []  # variables that dont learn anymore
        self.early = []
        self.late = []

    def train(self):
        self.trained_vars
        print("hi")



























