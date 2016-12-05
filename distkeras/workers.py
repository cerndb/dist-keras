"""
Workers module. This module contains all workers for the distributed
optimizers.
"""

## BEGIN Imports. ##############################################################

from distkeras.networking import *
from distkeras.utils import *

from itertools import tee

import time

import socket

import numpy as np

import zlib

## END Imports. ################################################################

class Worker(object):

    def __init__(self, model, optimizer, loss, features_col="features", label_col="label",
                 batch_size=32):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.features_column = features_col
        self.label_column = label_col
        self.batch_size = batch_size

    def prepare_model(self):
        # Deserialize the Keras model.
        self.model = deserialize_keras_model(self.model)
        # Compile the model with the specified loss and optimizer.
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])

    def train(self, worker_id, iterator):
        raise NotImplementedError

class SingleTrainerWorker(Worker):

    def __init__(self, model, optimizer, loss, features_col="features", label_col="label",
                 batch_size=32):
        # Initialize the parent class.
        super(SingleTrainerWorker, self).__init__(model, optimizer, loss, features_col,
                                                  label_col, batch_size)

    def train(self, worker_id, iterator):
        # Prepare the model.
        self.prepare_model()
        try:
            while True:
                # Fetch the next mini-batch.
                batch = [next(iterator) for _ in range(self.batch_size)]
                # Retrieve the feature and label vectors.
                feature_iterator, label_iterator = tee(batch, 2)
                X = np.asarray([x[self.features_column] for x in feature_iterator])
                Y = np.asarray([x[self.label_column] for x in label_iterator])
                self.model.train_on_batch(X, Y)
        except StopIteration:
            pass

        return iter([serialize_keras_model(self.model)])

class NetworkWorker(Worker):

    def __init__(self, model, optimizer, loss, features_col="features", label_col="label",
                 batch_size=32, master_host="localhost", master_port=5000):
        super(NetworkWorker, self).__init__(model, optimizer, loss, features_col,
                                            label_col, batch_size)
        self.master_host = master_host
        self.master_port = master_port
        self.worker_id = 0

    def set_worker_id(self, worker_id):
        self.worker_id = worker_id

    def get_worker_id(self):
        return self.worker_id

    def get_master_host(self):
        return self.master_host

    def get_master_port(self):
        return self.master_port

    def train(self, worker_id, iterator):
        raise NotImplementedError

class DOWNPOURWorker(NetworkWorker):

    def __init__(self, model, optimizer, loss, features_col="features", label_col="label",
                 batch_size=32, master_host="localhost", master_port=5000, learning_rate=0.01,
                 communication_window=3):
        # Initialize the parent object.
        super(DOWNPOURWorker, self).__init__(model, optimizer, loss, features_col, label_col,
                                             batch_size, master_host, master_port)
        # Initialize DOWNPOUR specific variables.
        self.learning_rate = learning_rate
        self.communication_window = communication_window
        self.iteration = 1

    def fetch_center_variable(self):
        cv = rest_get(self.master_host, self.master_port, '/center_variable')
        self.center_variable = np.asarray(cv)

    def send_residual(self, v):
        data = {}
        data['worker_id'] = self.get_worker_id()
        data['variable'] = v
        rest_post(self.master_host, self.master_port, '/update', data)

    def train(self, worker_id, iterator):
        # Prepare the model.
        self.prepare_model()
        # Set the worker id.
        self.set_worker_id(worker_id)
        # Uniformily initialize the replica with random weights.
        uniform_weights(self.model)
        # Prepare the gradient residual matrix.
        v = np.asarray(self.model.get_weights())
        v.fill(0.0)
        # Start the epoch training process.
        try:
            while True:
                # Fetch the next mini-batch.
                batch = [next(iterator) for _ in range(self.batch_size)]
                # Extract the feature and label vector.
                feature_iterator, label_iterator = tee(batch, 2)
                X = np.asarray([x[self.features_column] for x in feature_iterator])
                Y = np.asarray([x[self.label_column] for x in label_iterator])
                # Check if the residual needs to be communicated.
                if self.iteration % self.communication_window == 0:
                    # Send the residual to the master.
                    self.send_residual(v)
                    # Clear the residual
                    v.fill(0.0)
                    # Update the local variable.
                    self.fetch_center_variable()
                    # Update the local replica.
                    self.model.set_weights(self.center_variable)
                W1 = np.asarray(self.model.get_weights())
                self.model.train_on_batch(X, Y)
                W2 = np.asarray(self.model.get_weights())
                gradient = W2 - W1
                v += gradient
                self.iteration += 1
        except StopIteration:
            pass

        return iter([])

class DOWNPOURSocketWorker(NetworkWorker):

    def __init__(self, model, optimizer, loss, features_col="features", label_col="label",
                 batch_size=32, master_host="localhost", master_port=5000, learning_rate=0.01,
                 communication_window=3):
        # Initialize the parent object.
        super(DOWNPOURSocketWorker, self).__init__(model, optimizer, loss, features_col, label_col,
                                                   batch_size, master_host, master_port)
        # Initialize DOWNPOUR parameters.
        self.learning_rate = learning_rate
        self.communication_window = communication_window
        self.iteration = 1
        self.socket = None

    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.master_host, self.master_port))

    def pull(self):
        # Request a pull from the parameter server.
        self.socket.sendall(b'p')
        # Fetch the central variable from the parameter server.
        center_variable = recv_data(self.socket)
        self.center_variable = np.asarray(center_variable)

    def commit(self, delta):
        # Prepare the datastructure.
        data = {}
        data['worker_id'] = self.get_worker_id()
        data['delta'] = delta
        # Request a commit from the parameter server.
        self.socket.sendall(b'c')
        # Send the data to the parameter server.
        send_data(self.socket, data)

    def train(self, worker_id, iterator):
        # Prepare the model.
        self.prepare_model()
        # Uniformily initialize the replica with random weights.
        uniform_weights(self.model)
        # Connect to the parameter server.
        self.connect()
        # Set the worker id.
        self.set_worker_id(worker_id)
        # Prepare the gradient residual matrix.
        v = np.asarray(self.model.get_weights())
        v.fill(0.0)
        # Pull the current state of the center variable.
        self.pull()
        # Start the epoch training process
        try:
            # Fetch the next mini-batch.
            batch = [next(iterator) for _ in range(self.batch_size)]
            # Extract the feature and label vector.
            feature_iterator, label_iterator = tee(batch, 2)
            X = np.asarray([x[self.features_column] for x in feature_iterator])
            Y = np.asarray([x[self.label_column] for x in label_iterator])
            # Check if the residual needs to be communicated.
            if self.iteration % self.communication_window == 0:
                # Send the residual to the master.
                self.commit(v)
                # Clear the residual
                v.fill(0.0)
                # Update the local variable.
                self.pull()
                # Update the local replica.
                self.model.set_weights(self.center_variable)
            W1 = np.asarray(self.model.get_weights())
            self.model.train_on_batch(X, Y)
            W2 = np.asarray(self.model.get_weights())
            delta = W2 - W1
            v += delta
            self.iteration += 1
        except StopIteration:
            pass
        # Commit the last residual.
        self.commit(v)
        # Close the socket.
        self.socket.close()

        return iter([])

class EASGDWorker(NetworkWorker):

    def __init__(self, model, optimizer, loss, features_col="features", label_col="label",
                 batch_size=32, master_host="localhost", master_port=5000, rho=5.0,
                 learning_rate=0.01):
        # Initialize the parent object.
        super(EASGDWorker, self).__init__(model, optimizer, loss, features_col, label_col,
                                          batch_size, master_host, master_port)
        # Initialize EASGD specific variables.
        self.rho = rho
        self.learning_rate = learning_rate
        self.iteration = 1

    def send_variable(self, variable):
        data = {}
        data['worker_id'] = self.get_worker_id()
        data['variable'] = variable
        data['iteration'] = self.iteration
        rest_post(self.master_host, self.master_port, '/update', data)

    def master_ready(self):
        data = {}
        data['iteration'] = self.iteration
        master_ready = int(rest_post(self.master_host, self.master_port, '/ready', data))

        return master_ready == 1

    def fetch_center_variable(self):
        cv = rest_get(self.master_host, self.master_port, '/center_variable')
        self.center_variable = np.asarray(cv)

    def train(self, worker_id, iterator):
        # Prepare the model.
        self.prepare_model()
        # Set the worker id.
        self.set_worker_id(worker_id)
        # Start the epoch training.
        try:
            while True:
                # Fetch the center variable.
                self.fetch_center_variable()
                # Fetch the next mini-batch.
                batch = [next(iterator) for _ in range(self.batch_size)]
                # Extract the feature and label vector.
                feature_iterator, label_iterator = tee(batch, 2)
                X = np.asarray([x[self.features_column] for x in feature_iterator])
                Y = np.asarray([x[self.label_column] for x in label_iterator])
                W1 = np.asarray(self.model.get_weights())
                self.model.train_on_batch(X, Y)
                W2 = np.asarray(self.model.get_weights())
                gradient = W2 - W1
                self.send_variable(W2)
                W = W1 - self.learning_rate * (gradient + self.rho * (W1 - self.center_variable))
                self.model.set_weights(W)
                # Wait for the master to synchronize the workers.
                while not self.master_ready():
                    time.sleep(0.1)
                self.iteration += 1
        except StopIteration:
            pass

        return iter([])

class AEASGDWorker(NetworkWorker):

    def __init__(self, model, optimizer, loss, features_col="features", label_col="label",
                 batch_size=32, master_host="localhost", master_port=5000, rho=5.0,
                 learning_rate=0.01, communication_window=32):
        # Initialize the parent object.
        super(AEASGDWorker, self).__init__(model, optimizer, loss, features_col, label_col,
                                           batch_size, master_host, master_port)
        # Initialize AEASGD specific variables.
        self.rho = rho
        self.learning_rate = learning_rate
        self.communication_window = communication_window
        self.alpha = self.rho * self.learning_rate
        self.iteration = 1

    def fetch_center_variable(self):
        cv = rest_get(self.master_host, self.master_port, '/center_variable')
        self.center_variable = np.asarray(cv)

    def send_elastic_difference(self, ed):
        data = {}
        data['worker_id'] = self.get_worker_id()
        data['variable'] = ed
        rest_post(self.master_host, self.master_port, '/update', data)

    def train(self, worker_id, iterator):
        # Prepare the model.
        self.prepare_model()
        # Set the worker id.
        self.set_worker_id(worker_id)
        # Start the epoch training.
        try:
            while True:
                # Fetch the next mini-batch.
                batch = [next(iterator) for _ in range(self.batch_size)]
                # Extract the feature and label vector.
                feature_iterator, label_iterator = tee(batch, 2)
                X = np.asarray([x[self.features_column] for x in feature_iterator])
                Y = np.asarray([x[self.label_column] for x in label_iterator])
                # Check if we need to communicate with the parameter server.
                if self.iteration % self.communication_window == 0:
                    self.fetch_center_variable()
                    W = np.asarray(self.model.get_weights())
                    E = self.alpha * (W - self.center_variable)
                    W = W - E
                    self.model.set_weights(W)
                    self.send_elastic_difference(E)
                self.model.train_on_batch(X, Y)
                self.iteration += 1
        except StopIteration:
            pass

        return iter([])

class EAMSGDWorker(NetworkWorker):

    def __init__(self, model, optimizer, loss, features_col="features", label_col="label",
                 batch_size=32, master_host="localhost", master_port=5000, rho=5.0,
                 learning_rate=0.01, momentum=0.9, communication_window=32):
        # Initialize the parent object.
        super(EAMSGDWorker, self).__init__(model, optimizer, loss, features_col, label_col,
                                           batch_size, master_host, master_port)
        # Initialize EAMSGD specific variables.
        self.rho = rho
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.communication_window = communication_window
        self.alpha = self.learning_rate * self.rho
        self.iteration = 1

    def fetch_center_variable(self):
        cv = rest_get(self.master_host, self.master_port, '/center_variable')
        self.center_variable = np.asarray(cv)

    def send_elastic_difference(self, ed):
        data = {}
        data['worker_id'] = self.get_worker_id()
        data['variable'] = ed
        rest_post(self.master_host, self.master_port, '/update', data)

    def train(self, worker_id, iterator):
        # Prepare the model.
        self.prepare_model()
        # Set the worker identifier.
        self.set_worker_id(worker_id)
        # Initialize the momentum residual matrix.
        v = np.asarray(self.model.get_weights())
        v.fill(0.0)
        # Start the epoch training.
        try:
            while True:
                # Fetch the next mini-batch.
                batch = [next(iterator) for _ in range(self.batch_size)]
                # Extract the feature and label vector.
                feature_iterator, label_iterator = tee(batch, 2)
                X = np.asarray([x[self.features_column] for x in feature_iterator])
                Y = np.asarray([x[self.label_column] for x in label_iterator])
                # Check if we need to communicate with the parameter server.
                if self.iteration % self.communication_window == 0:
                    # Update the local worker with the center variable.
                    self.fetch_center_variable()
                    W = np.asarray(self.model.get_weights())
                    # Compute the elastic difference.
                    E = self.alpha * (W - self.center_variable)
                    W = W - E
                    # Update the local replica.
                    self.model.set_weights(W)
                    # Send the elastic difference to the master.
                    self.send_elastic_difference(E)
                # Update the momentum residual.
                v_t = self.momentum * v
                W_copy = np.asarray(self.model.get_weights())
                W = np.asarray(self.model.get_weights())
                W += v_t
                self.model.set_weights(W)
                self.model.train_on_batch(X, Y)
                gradient = np.asarray(self.model.get_weights()) - W
                v = v_t - self.learning_rate * gradient
                W_copy -= v
                self.model.set_weights(W_copy)
                self.iteration += 1
        except StopIteration:
            pass

        return iter([])
