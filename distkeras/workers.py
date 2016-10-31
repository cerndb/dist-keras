"""
Workers module. This module contains all workers for the distributed
optimizers.
"""

## BEGIN Imports. ##############################################################

from distkeras.networking import *
from distkeras.utils import *

from itertools import tee

import numpy as np

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
                # Fetch the next batch.
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

class EAMSGDWorker(NetworkWorker):

    def __init__(self, model, optimizer, loss, features_col="features", label_col="label",
                 batch_size=32, master_host="localhost", master_port=5000, rho=5.0,
                 learning_rate=0.01, momentum=0.9, communication_window=10):
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
        # Initialize the momentum residual matrix.
        v = np.asarray(self.model.get_weights())
        v.fill(0.0)
        # Start the epoch training.
        try:
            while True:
                # Fetch the next batch.
                batch = [next(iterator) for _ in range(self.batch_size)]
                # Extract the feature and label vector.
                feature_iterator, label_iterator = tee(batch, 2)
                X = np.asarray(x[self.features_column] for x in feature_iterator)
                Y = np.asarray(x[self.label_column] for x in label_iterator)
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
                    self.send_elastic_difference(worker_id, E)
                # Update the momentum residual.
                v_t = self.momentum * v
                W_copy = np.asarray(self.model.get_weights())
                W = np.asarray(self.model.get_weights())
                W += v_t
                self.model.set_weights(W)
                model.train_on_batch(X, Y)
                gradient = np.asarray(model.get_weights()) - W
                v = v_t - self.learning_rate * gradient
                W_copy -= v
                self.model.set_weights(W_copy)
                self.iteration += 1
        except StopIteration:
            pass

        return iter([])
