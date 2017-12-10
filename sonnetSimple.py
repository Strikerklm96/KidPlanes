

import sonnet as snt

class FLAGS:
    output_size = 1

def get_training_data():
    return snt. .tensor([[0], [3]])

def get_test_data():
    return get_training_data()

# Provide your own functions to generate data Tensors.
train_data = get_training_data()
test_data = get_test_data()

# Construct the module, providing any configuration necessary.
linear_regression_module = snt.Linear(output_size=FLAGS.output_size)

# Connect the module to some inputs, any number of times.
train_predictions = linear_regression_module(train_data)
test_predictions = linear_regression_module(test_data)
