"""Utility functions used throughout Distributed Keras."""

## BEGIN Import. ###############################################################

from keras import backend as K

from keras.models import model_from_json

from keras import backend as K

from pyspark.mllib.linalg import DenseVector
from pyspark.sql import Row
from pyspark.sql.functions import rand

import pickle

import json

import numpy as np

import os

import pwd

## END Import. #################################################################


def get_os_username():
    """Returns the username of user on the operating system.

    From: http://stackoverflow.com/questions/842059/is-there-a-portable-way-to-get-the-current-username-in-python
    """
    return pwd.getpwuid(os.getuid())[0]


def set_keras_base_directory(base_dir='/tmp/' + get_os_username()):
    """Sets the base directory of Keras."""
    K._keras_base_dir = base_dir


def to_one_hot_encoded_dense(value, n_dim=2):
    """Converts the value to a one-hot encoded vector.

    # Arguments
        value: float. Value of the single "hot" value.
        n_dim: int. Dimension of the output vector.
    """
    value = int(value)
    vector = np.zeros(n_dim)
    vector[value] = 1.0

    return vector


def new_dataframe_row(old_row, column_name, column_value):
    """Constructs a new Spark Row based on the old row, and a new column name and value."""
    row = Row(*(old_row.__fields__ + [column_name]))(*(old_row + (column_value, )))

    return row


def json_to_dataframe_row(string):
    """Converts a JSON String to a Spark Dataframe row."""
    dictionary = json.loads(string)
    row = Row(**dictionary)

    return row


def pickle_object(o):
    """Pickles the specified model and its weights."""
    return pickle.dumps(o, -1)


def unpickle_object(string):
    """Unpickles the specified string into a model."""
    return pickle.loads(string)


def serialize_keras_model(model):
    """Serializes the specified Keras model into a dictionary."""
    dictionary = {}
    dictionary['model'] = model.to_json()
    dictionary['weights'] = model.get_weights()

    return dictionary


def history_executors_average(history):
    """Returns the averaged training metrics for all the executors."""
    max_iteration = max(history, key=lambda x: x['iteration'])['iteration']
    max_executor = max(history, key=lambda x: x['worker_id'])['worker_id']
    histories = []
    averaged_history = []
    # Fetch the histories of the individual executors.
    for i in range(0, max_executor):
        histories.append(history_executor(history, i))
    # Construct the averaged history.
    for i in range(0, max_iteration):
        num_executors = 0
        sum = np.zeros(2)
        for j in range(0, max_executor):
            if len(histories[j]) - 1 >= i:
                num_executors += 1
                sum += histories[j][i]['history']
        # Average the history.
        sum /= num_executors
        averaged_history.append(sum)

    return averaged_history


def history_executor(history, id):
    """Returns the history of a specific executor."""
    executor_history = [h for h in history if h['worker_id'] == id]
    executor_history.sort(key=lambda x: x['iteration'])

    return executor_history


def deserialize_keras_model(dictionary):
    """Deserialized the Keras model using the specified dictionary."""
    architecture = dictionary['model']
    weights = dictionary['weights']
    model = model_from_json(architecture)
    model.set_weights(weights)

    return model


def uniform_weights(model, constraints=[-0.5, 0.5]):
    """Initializes the parameters of the specified Keras model with uniform
    weights between the specified ranges.

    # Arguments
        model: Keras model.
        constraints: array. An array with two elements which defines the range
                     of the uniform initalization.
    """
    # We assume the following: Keras will return a list of weight matrices.
    # All layers, even the activiation layers, will be randomly initialized.
    weights = model.get_weights()
    for layer in weights:
        shape = layer.shape
        if len(shape) > 1:
            # Fill the matrix with random numbers.
            n_rows = shape[0]
            n_columns = shape[1]
            for i in range(0, n_rows):
                for j in range(0, n_columns):
                    layer[i][j] = np.random.uniform(low=constraints[0], high=constraints[1])
        else:
            # Fill the vector with random numbers.
            n_elements = shape[0]
            for i in range(0, n_elements):
                layer[i] = np.random.uniform(low=constraints[0], high=constraints[1])
    # Set the new weights in the model.
    model.set_weights(weights)


def shuffle(dataset):
    """Shuffles the rows in the specified Spark Dataframe.

    # Arguments
        dataset: dataframe. A Spark Dataframe.
    """
    dataset = dataset.orderBy(rand())
    dataset.cache()

    return dataset


def precache(dataset, num_workers):
    """Precaches the specified dataset.

    Make sure the specified dataframe has the desired partitioning scheme.

    # Arguments
        dataset: dataframe. A Spark Dataframe.
        num_workers: int. Number of workers you are going to use.
    """
    dataset = dataset.repartition(num_workers)
    dataset.cache()
    dataset.count()

    return dataset
