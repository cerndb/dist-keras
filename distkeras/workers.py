"""
Workers module. This module contains all workers for the distributed
optimizers.
"""

## BEGIN Imports. ##############################################################

from distkeras.networking import *
from distkeras.utils import *

from itertools import tee

from keras.engine.training import make_batches
from keras.engine.training import slice_X

import numpy as np

import time

## END Imports. ################################################################

class EASGDWorker(object):

    def __init__(self, keras_model, worker_optimizer, loss, features_col="features", label_col="label",
                 batch_size=32, rho=5, learning_rate=0.01, master_host="localhost", master_port=5000):
        self.model = keras_model
        self.features_column = features_col
        self.optimizer = worker_optimizer
        self.loss = loss
        self.label_column = label_col
        self.master_host = master_host
        self.master_port = master_port
        self.center_variable = None
        self.batch_size = batch_size
        self.rho = rho
        self.iteration = 1
        self.learning_rate = learning_rate

    def master_send_variable(self, worker_id, variable, history):
        data = {}
        data['worker_id'] = worker_id
        data['iteration'] = self.iteration
        data['variable'] = variable
        data['history'] = history
        rest_post(self.master_host, self.master_port, "/update", data)

    def master_is_ready(self):
        data = {}
        data['iteration'] = self.iteration
        master_ready = int(rest_post(self.master_host, self.master_port, "/ready", data))

        return master_ready == 1

    def fetch_center_variable(self):
        self.center_variable = np.asarray(rest_get(self.master_host, self.master_port, "/center_variable"))

    def train(self, index, iterator):
        model = deserialize_keras_model(self.model)
        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=['accuracy'])
        try:
            while True:
                self.fetch_center_variable()
                batch = [next(iterator) for _ in range(self.batch_size)]
                feature_iterator, label_iterator = tee(batch, 2)
                X = np.asarray([x[self.features_column] for x in feature_iterator])
                Y = np.asarray([x[self.label_column] for x in label_iterator])
                W1 = np.asarray(model.get_weights())
                history = model.train_on_batch(X, Y)
                W2 = np.asarray(model.get_weights())
                gradient = W2 - W1
                self.master_send_variable(index, W2, history)
                W = W1 - self.learning_rate * (gradient + self.rho * (W1 - self.center_variable))
                model.set_weights(W)
                while not self.master_is_ready():
                    time.sleep(0.2)
                self.iteration += 1
        except StopIteration:
            pass

        return iter([])

class AsynchronousEAMSGDWorker(object):

    def __init__(self, keras_model, worker_optimizer, loss, features_col="features",
                 label_col="label", batch_size=32, rho=5.0, learning_rate=0.01, master_host="localhost",
                 master_port=5000, communication_window=10, momentum=0.95):
        self.model = keras_model
        self.features_column = features_col
        self.label_column = label_col
        self.optimizer = worker_optimizer
        self.loss = loss
        self.master_host = master_host
        self.master_port = master_port
        self.center_variable = None
        self.batch_size = batch_size
        self.rho = rho
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.iteration = 1
        self.communication_window = communication_window
        self.alpha = self.learning_rate * self.rho

    def master_send_ed(self, worker_id, variable):
        data = {}
        data['worker_id'] = worker_id
        data['iteration'] = self.iteration
        data['variable'] = variable
        rest_post(self.master_host, self.master_port, "/update", data)

    def fetch_center_variable(self):
        c_v = rest_get(self.master_host, self.master_port, "/center_variable")
        self.center_variable = np.asarray(c_v)

    def train(self, index, iterator):
        model = deserialize_keras_model(self.model)
        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=['accuracy'])
        # Initialize the residual weight matrices.
        v = np.asarray(model.get_weights())
        v.fill(0.0)
        try:
            while True:
                batch = [next(iterator) for _ in range(self.batch_size)]
                feature_iterator, label_iterator = tee(batch, 2)
                X = np.asarray([x[self.features_column] for x in feature_iterator])
                Y = np.asarray([x[self.label_column] for x in label_iterator])
                if self.iteration % self.communication_window == 0:
                    self.fetch_center_variable()
                    W = np.asarray(model.get_weights())
                    # Compute the elastic difference.
                    E = self.alpha * (W - self.center_variable)
                    W = W - E
                    model.set_weights(W)
                    # Send the elastic difference to the master.
                    self.master_send_ed(index, E)
                v_t = self.momentum * v
                W_backup = np.asarray(model.get_weights())
                W = np.asarray(model.get_weights())
                W += v_t
                model.set_weights(W)
                model.train_on_batch(X, Y)
                gradient = np.asarray(model.get_weights()) - W
                v = v_t - self.learning_rate * gradient
                W_backup -= v
                model.set_weights(W_backup)
                self.iteration += 1
        except StopIteration:
            pass

        return iter([])

class AsynchronousEASGDWorker(object):

    def __init__(self, keras_model, worker_optimizer, loss, features_col="features", label_col="label",
                 batch_size=32, rho=5.0, learning_rate=0.01, master_host="localhost",
                 master_port=5000, communication_window=5):
        self.model = keras_model
        self.features_column = features_col
        self.optimizer = worker_optimizer
        self.loss = loss
        self.label_column = label_col
        self.master_host = master_host
        self.master_port = master_port
        self.center_variable = None
        self.batch_size = batch_size
        self.rho = rho
        self.learning_rate = learning_rate
        self.alpha = self.learning_rate * self.rho
        self.communication_period = communication_window
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
        model = deserialize_keras_model(self.model)
        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
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
                    W = W - E
                    model.set_weights(W)
                    # Send the elastic difference to the master.
                    self.master_send_ed(index, E)
                model.train_on_batch(X, Y)
                self.iteration += 1
        except StopIteration:
            pass

        return iter([])

class DOWNPOURWorker(object):

    def __init__(self, keras_model, worker_optimizer, loss, features_col="features", label_col="label",
                 batch_size=1000, master_host="localhost", learning_rate=0.01,
                 master_port=5000, communication_window=5):
        self.model = keras_model
        self.features_column = features_col
        self.optimizer = worker_optimizer
        self.loss = loss
        self.label_column = label_col
        self.master_host = master_host
        self.master_port = master_port
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.communication_window = communication_window
        self.iteration = 1
        self.center_variable = None

    def master_send_v(self, worker_id, v):
        data = {}
        data['worker_id'] = worker_id
        data['variable'] = v
        data['iteration'] = self.iteration
        rest_post(self.master_host, self.master_port, '/update', data)

    def fetch_center_variable(self):
        self.center_variable = np.asarray(rest_get(self.master_host, self.master_port, "/center_variable"))

    def train(self, index, iterator):
        model = deserialize_keras_model(self.model)
        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=['accuracy'])
        uniform_weights(model)
        v = np.asarray(model.get_weights())
        v.fill(0.0)
        try:
            while True:
                batch = [next(iterator) for _ in range(self.batch_size)]
                feature_iterator, label_iterator = tee(batch, 2)
                X = np.asarray([x[self.features_column] for x in feature_iterator])
                Y = np.asarray([x[self.label_column] for x in label_iterator])
                if self.iteration % self.communication_window == 0:
                    # Send the variable to the parameter server, and reset.
                    self.master_send_v(index, v)
                    v.fill(0.0)
                    # Update the local variable.
                    self.fetch_center_variable()
                    model.set_weights(self.center_variable)
                W1 = np.asarray(model.get_weights())
                model.train_on_batch(X, Y)
                W2 = np.asarray(model.get_weights())
                # Update the distributed variable
                gradient = (W2 - W1)
                v += gradient
                self.iteration += 1
        except StopIteration:
            pass

        return iter([])

class SingleTrainerWorker(object):

    def __init__(self, keras_model, worker_optimizer, loss, features_col="features", label_col="label",
                 batch_size=32):
        self.model = keras_model
        self.features_column = features_col
        self.loss = loss
        self.optimizer = worker_optimizer
        self.label_column = label_col
        self.batch_size = batch_size

    def train(self, iterator):
        model = deserialize_keras_model(self.model)
        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=['accuracy'])
        try:
            while True:
                batch = [next(iterator) for _ in range(self.batch_size)]
                feature_iterator, label_iterator = tee(batch, 2)
                X = np.asarray([x[self.features_column] for x in feature_iterator])
                Y = np.asarray([x[self.label_column] for x in label_iterator])
                model.train_on_batch(X, Y)
        except StopIteration:
            pass

        return iter([serialize_keras_model(model)])
