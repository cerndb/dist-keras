"""
Workers module. This module contains all workers for the distributed
optimizers.
"""

## BEGIN Imports. ##############################################################

from distkeras.networking import *
from distkeras.utils import *

from itertools import tee

from keras.optimizers import Adagrad

import numpy as np

import time

## END Imports. ################################################################

class EASGDWorker(object):

    def __init__(self, keras_model, features_col="features", label_col="label", batch_size=1000,
                 rho=5, learning_rate=0.01, master_host="localhost", master_port=5000):
        self.model = keras_model
        self.features_column = features_col
        self.label_column = label_col
        self.master_host = master_host
        self.master_port = master_port
        self.center_variable = None
        self.batch_size = batch_size
        self.rho = rho
        self.iteration = 1
        self.learning_rate = learning_rate

    def master_send_variable(self, worker_id, variable):
        data = {}
        data['worker_id'] = worker_id
        data['iteration'] = self.iteration
        data['variable'] = variable
        rest_post(self.master_host, self.master_port, "/update", data)

    def master_is_ready(self):
        data = {}
        data['iteration'] = self.iteration
        master_ready = int(rest_post(self.master_host, self.master_port, "/ready", data))

        return master_ready == 1

    def fetch_center_variable(self):
        self.center_variable = np.asarray(rest_get(self.master_host, self.master_port, "/center_variable"))

    def train(self, index, iterator):
        # Deserialize the Keras model.
        model = deserialize_keras_model(self.model)
        # Initialize the model weights with a constrainted uniform distribution.
        #weights = uniform_weights(model.get_weights(), [-5, 5])
        #model.set_weights(weights)
        # Compile the model.
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adagrad(),
                      metrics=['accuracy'])
        try:
            while True:
                self.fetch_center_variable()
                batch = [next(iterator) for _ in range(self.batch_size)]
                feature_iterator, label_iterator = tee(batch, 2)
                X = np.asarray([x[self.features_column] for x in feature_iterator])
                Y = np.asarray([x[self.label_column] for x in label_iterator])
                W1 = np.asarray(model.get_weights())
                model.fit(X, Y, nb_epoch=1)
                W2 = np.asarray(model.get_weights())
                gradient = W2 - W1
                self.master_send_variable(index, W2)
                W = W1 - self.learning_rate * (gradient + self.rho * (W1 - self.center_variable))
                model.set_weights(W)
                while not self.master_is_ready():
                    time.sleep(0.2)
                self.iteration += 1
        except StopIteration:
            pass

        return iter([])

class AsynchronousEASGDWorker(object):

    def __init__(self, keras_model, features_col="features", label_col="label",
                 batch_size=1000, rho=5.0, learning_rate=0.01, master_host="localhost",
                 master_port=5000, communication_window=5, nb_epoch=1):
        self.model = keras_model
        self.features_column = features_col
        self.label_column = label_col
        self.master_host = master_host
        self.master_port = master_port
        self.center_variable = None
        self.batch_size = batch_size
        self.rho = rho
        self.learning_rate = learning_rate
        self.alpha = self.learning_rate * self.rho
        self.communication_period = communication_window
        self.nb_epoch = nb_epoch
        self.iteration = 1

    def master_send_ed(self, worker_id, variable):
        data = {}
        data['worker_id'] = worker_id
        data['iteration'] = self.iteration
        data['variable'] = variable
        rest_post(self.master_host, self.master_port, "/update", data)

    def fetch_center_variable(self):
        self.center_variable = np.asarray(rest_get(self.master_host, self.master_port, "/center_variable"))

    def train(self, index, iterator):
        # Deserialize the Keras model.
        model = deserialize_keras_model(self.model)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adagrad(),
                      metrics=['accuracy'])
        try:
            while True:
                batch = [next(iterator) for _ in range(self.batch_size)]
                feature_iterator, label_iterator = tee(batch, 2)
                X = np.asarray([x[self.features_column] for x in feature_iterator])
                Y = np.asarray([x[self.label_column] for x in label_iterator])
                if self.iteration % self.communication_period == 0:
                    self.fetch_center_variable()
                    W = np.asarray(model.get_weights())
                    # Compute the elastic difference.
                    E = self.alpha * (W - self.center_variable)
                    # Update the model.
                    W = W - E
                    model.set_weights(W)
                    # Sent the elastic difference back to the master.
                    self.master_send_ed(index, E)
                model.fit(X, Y, nb_epoch=1)
                self.iteration += 1
        except StopIteration:
            pass

        return iter([])

class EnsembleTrainerWorker(object):

    def __init__(self, keras_model, features_col="features", label_col="label", label_transformer=None):
        self.model = keras_model
        self.features_column = features_col
        self.label_column = label_col
        self.label_transformer = label_transformer

    def train(self, iterator):
        # Deserialize the Keras model.
        model = deserialize_keras_model(self.model)
        feature_iterator, label_iterator = tee(iterator, 2)
        X = np.asarray([x[self.features_column] for x in feature_iterator])
        # Check if a label transformer is available.
        if self.label_transformer:
            Y = np.asarray([self.label_transformer(x[self.label_column]) for x in label_iterator])
        else:
            Y = np.asarray([x[self.label_column] for x in label_iterator])
        # TODO Add compilation parameters.
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adagrad(),
                      metrics=['accuracy'])
        # Fit the model with the data.
        history = model.fit(X, Y, nb_epoch=1)
        partitionResult = (history, model)

        return iter([partitionResult])

class SingleTrainerWorker(object):

    def __init__(self, keras_model, features_col="features", label_col="label", batch_size=1000, num_epoch=1):
        self.model = keras_model
        self.features_column = features_col
        self.label_column = label_col
        self.batch_size = batch_size
        self.num_epoch = num_epoch

    def train(self, iterator):
        model = deserialize_keras_model(self.model)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adagrad(),
                      metrics=['accuracy'])
        training_time = 0
        try:
            while True:
                batch = [next(iterator) for _ in range(self.batch_size)]
                feature_iterator, label_iterator = tee(batch, 2)
                X = np.asarray([x[self.features_column] for x in feature_iterator])
                Y = np.asarray([x[self.label_column] for x in label_iterator])
                start_time = time.time()
                model.fit(X, Y, nb_epoch=self.num_epoch)
                training_time += time.time() - start_time
        except StopIteration:
            pass

        print("Training time: " + `training_time`)

        return iter([serialize_keras_model(model)])

class DGEWorker(object):

    def __init__(self, keras_model, features_col="features", label_col="label", batch_size=1000,
                 master_host="localhost", master_port=5000):
        self.model = keras_model
        self.features_column = features_col
        self.label_column = label_col
        self.batch_size = batch_size
        self.master_host = master_host
        self.master_port = master_port
        self.iteration = 1
