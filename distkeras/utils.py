"""
Utils module.
"""

## BEGIN Import. ###############################################################

from pyspark.sql import Row
from pyspark.mllib.linalg import DenseVector
from pyspark.sql.functions import rand
from pyspark.sql.functions import udf
from pyspark.sql.types import *

from keras.models import model_from_json

from itertools import izip_longest

import numpy as np

## END Import. #################################################################

def to_dense_vector(value, n_dim=2):
    vector = np.zeros(n_dim)
    vector[value] = 1.0

    return DenseVector(vector)

def new_dataframe_row(old_row, column_name, column_value):
    r = Row(*(old_row.__fields__ + [column_name]))(*(old_row + (column_value, )))

    return r

def serialize_keras_model(model):
    d = {}
    d['model'] = model.to_json()
    d['weights'] = model.get_weights()

    return d

def deserialize_keras_model(d):
    architecture = d['model']
    weights = d['weights']
    model = model_from_json(architecture)
    model.set_weights(weights)

    return model

def uniform_weights(model, constraints=[-0.5, 0.5]):
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

def weights_mean(weights):
    assert(weights.shape[0] > 1)

    return np.mean(weights, axis=0)

def weights_mean_vector(weights):
    num_weights = weights.shape[0]

    # Check if the precondition has been met.
    assert(num_weights > 1)

    w = []
    for weight in weights:
        flat = np.asarray([])
        for layer in weight:
            layer = layer.flatten()
            flat = np.hstack((flat, layer))
        w.append(flat)
    w = np.asarray(w)

    return np.mean(w, axis=0)

def weights_std(weights):
    num_weights = weights.shape[0]

    # Check if the precondition has been met.
    assert(num_weights > 1)

    w = []
    for weight in weights:
        flat = np.asarray([])
        for layer in weight:
            layer = layer.flatten()
            flat = np.hstack((flat, layer))
        w.append(flat)
    w = np.asarray(w)
    std = np.std(w, axis=0)
    for i in range(0, std.shape[0]):
        if std[i] == 0.0:
            std[i] = 0.000001

    return std

def shuffle(dataset):
    dataset = dataset.orderBy(rand())

    return dataset

def batches(iterable, n):
    batch = []
    size = len(iterable)
    for i in range(0, size, n):
        batch.append(iterable[i:min(i + n, 1)])

    return np.asarray(batch)

def get_weight_matrix_indices(model):
    matrices = np.asarray(model.get_weights())
    num_matrices = len(matrices)
    indices = []
    for i in range(0, num_matrices):
        if matrices[i].any():
            indices.append(i)

    return indices

def vectorize(model, weight_indices):
    matrices = model.get_weights()
    v = np.array([])
    for i in weight_indices:
        flat = matrices[i].flatten()
        v = np.hstack((v, flat))

    return v
