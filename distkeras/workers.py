"""Workers module.

This module contains all worker specific implementations for different optimization
algorithms.
"""

## BEGIN Imports. ##############################################################

from itertools import tee

import socket

import numpy as np

from multiprocessing import Pool

from distkeras.networking import send_data
from distkeras.networking import recv_data
from distkeras.utils import deserialize_keras_model
from distkeras.utils import serialize_keras_model
from distkeras.utils import shuffle
from distkeras.utils import uniform_weights

## END Imports. ################################################################

class Worker(object):
    """Abstract class of a worker.

    This class provides basic functionality and properties all workers share.
    """

    def __init__(self, model, optimizer, loss, features_col="features", label_col="label",
                 batch_size=32):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.features_column = features_col
        self.label_column = label_col
        self.batch_size = batch_size

    def prepare_model(self):
        """Prepares the model for training."""
        # Deserialize the Keras model.
        self.model = deserialize_keras_model(self.model)
        # Compile the model with the specified loss and optimizer.
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])

    def train(self, worker_id, iterator):
        """Training procedure for the worker node.

        # Arguments
            worker_id: int. Partition index provided by Spark. Can be used as a worker_id.
            iterator: iterator. Data iterator.
        """
        raise NotImplementedError


class SequentialWorker(Worker):
    """Implementation for sequential gradient updates on a single worker.

    Will train a model on a single worker node.
    """

    def __init__(self, model, optimizer, loss, features_col="features", label_col="label",
                 batch_size=32):
        # Initialize the parent class.
        super(SequentialWorker, self).__init__(model, optimizer, loss, features_col,
                                               label_col, batch_size)

    def train(self, worker_id, iterator):
        """Training procedure with sequential gradient updates.

        # Returns
            Trained serialized Keras model.
        """
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
    """Abstract class of a worker who shares the variables using the network."""

    def __init__(self, model, optimizer, loss, features_col="features", label_col="label",
                 batch_size=32, master_host="localhost", master_port=5000):
        super(NetworkWorker, self).__init__(model, optimizer, loss, features_col,
                                            label_col, batch_size)
        self.master_host = master_host
        self.master_port = master_port
        self.worker_id = 0

    def set_worker_id(self, worker_id):
        """Sets the worker id.

        # Arguments
            worker_id: int. Worker identifier.
        """
        self.worker_id = worker_id

    def get_worker_id(self):
        """Returns the worker id."""
        return self.worker_id

    def get_master_host(self):
        """Returns the host address of the master parameter server."""
        return self.master_host

    def get_master_port(self):
        """Returns the port of the master parameter server."""
        return self.master_port

    def train(self, worker_id, iterator):
        """Abstract training procedure of a network based trainer.

        See: distkeras.workers.Worker.train
        """
        raise NotImplementedError


class MassWorker(NetworkWorker):
    """Experimental optimization algorithm."""

    def __init__(self, model, optimizer, loss, features_col="features", label_col="label",
                 batch_size=32, master_host="localhost", master_port=5000, learning_rate=0.01):
        # Initialize the parent object.
        super(MassWorker, self).__init__(model, optimizer, loss, features_col, label_col,
                                         batch_size, master_host, master_port)
        # Initialize Mass parameters.
        self.learning_rate = learning_rate
        self.iteration = 1
        self.socket = None
        self.center_variable = None

    def connect(self):
        """Connects with the parameter server."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.master_host, self.master_port))

    def pull(self):
        """Requests the center variable from the parameter server."""
        # Request a pull from the parameter server.
        self.socket.sendall(b'p')
        # Fetch the center variable from the parameter server.
        center_variable = recv_data(self.socket)
        self.center_variable = np.asarray(center_variable)

    def commit(self, delta):
        """Commits the delta to the parameter server."""
        # Prepare the data structure.
        data = {}
        data['worker_id'] = self.get_worker_id()
        data['delta'] = delta
        # Request a commit from the parameter server.
        self.socket.sendall(b'c')
        # Send the data to the parameter server.
        send_data(self.socket, data)

    def train(self, worker_id, iterator):
        """Training procedure for the Mass optimizer."""
        # Prepare the model.
        self.prepare_model()
        # Connect to the parameter server.
        self.connect()
        # Set the worker id.
        self.set_worker_id(worker_id)
        # Prepare the gradient residual matrix.
        v = np.asarray(self.model.get_weights(), dtype=np.float32)
        v.fill(0.0)
        # Start the epoch training process
        try:
            while True:
                # Fetch the next mini-batch.
                batch = [next(iterator) for _ in range(self.batch_size)]
                # Extract the feature and label vector.
                feature_iterator, label_iterator = tee(batch, 2)
                X = np.asarray([x[self.features_column] for x in feature_iterator])
                Y = np.asarray([x[self.label_column] for x in label_iterator])
                # Check if the residual needs to be communicated.
                if self.iteration % 5 == 0:
                    # Send the residual to the master.
                    self.commit(v)
                    # Clear the residual
                    v.fill(0.0)
                    # Update the local variable.
                    self.pull()
                    # Update the local replica.
                    self.model.set_weights(self.center_variable)
                W1 = np.asarray(self.model.get_weights(), dtype=np.float32)
                self.model.train_on_batch(X, Y)
                W2 = np.asarray(self.model.get_weights(), dtype=np.float32)
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


class DOWNPOURWorker(NetworkWorker):
    """Implements the training procedure for the distributed DOWNPOUR optimizer.

    Introduced by Dean et al.
    http://static.googleusercontent.com/media/research.google.com/en//archive/large_deep_networks_nips2012.pdf
    """

    def __init__(self, model, optimizer, loss, features_col="features", label_col="label",
                 batch_size=32, master_host="localhost", master_port=5000, learning_rate=0.01,
                 communication_window=3):
        # Initialize the parent object.
        super(DOWNPOURWorker, self).__init__(model, optimizer, loss, features_col, label_col,
                                             batch_size, master_host, master_port)
        # Initialize DOWNPOUR parameters.
        self.learning_rate = learning_rate
        self.communication_window = communication_window
        self.iteration = 1
        self.socket = None
        self.center_variable = None

    def connect(self):
        """Connects with the parameter server."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.master_host, self.master_port))

    def pull(self):
        """Requests the center variable from the parameter server."""
        # Request a pull from the parameter server.
        self.socket.sendall(b'p')
        # Fetch the central variable from the parameter server.
        center_variable = recv_data(self.socket)
        self.center_variable = np.asarray(center_variable)

    def commit(self, delta):
        """Commits the delta to the parameter server."""
        # Prepare the datastructure.
        data = {}
        data['worker_id'] = self.get_worker_id()
        data['delta'] = delta
        # Request a commit from the parameter server.
        self.socket.sendall(b'c')
        # Send the data to the parameter server.
        send_data(self.socket, data)

    def train(self, worker_id, iterator):
        """Specific training procedure for DOWNPOUR."""
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
        # Start the epoch training process
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
                    self.commit(v)
                    # Clear the residual
                    v.fill(0.0)
                    # Update the local variable.
                    self.pull()
                    # Update the local replica.
                    self.model.set_weights(self.center_variable)
                W1 = np.asarray(self.model.get_weights(), dtype=np.float32)
                self.model.train_on_batch(X, Y)
                W2 = np.asarray(self.model.get_weights(), dtype=np.float32)
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


class AEASGDWorker(NetworkWorker):
    """Implementation of asynchronous EASGD worker.

    Introduced by Zhang et al.
    https://arxiv.org/pdf/1412.6651.pdf
    """

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
        self.socket = None
        self.center_variable = None

    def connect(self):
        """Connects with the parameter server."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.master_host, self.master_port))

    def pull(self):
        """Requests the center variable from the parameter server."""
        # Request a pull from the parameter server.
        self.socket.sendall(b'p')
        # Fetch the central variable from the parameter server.
        center_variable = recv_data(self.socket)
        self.center_variable = np.asarray(center_variable)

    def commit(self, delta):
        """Sends the delta to the parameter server."""
        # Prepare the datastructure.
        data = {}
        data['worker_id'] = self.get_worker_id()
        data['delta'] = delta
        # Request a commit from the parameter server.
        self.socket.sendall(b'c')
        # Send the data to the parameter server.
        send_data(self.socket, data)

    def train(self, worker_id, iterator):
        """Specific training procedure for AEASGD."""
        # Prepare the model.
        self.prepare_model()
        # Set the worker id.
        self.set_worker_id(worker_id)
        # Connect to the parameter server.
        self.connect()
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
                    self.pull()
                    W = np.asarray(self.model.get_weights())
                    E = self.alpha * (W - self.center_variable)
                    W = W - E
                    self.model.set_weights(W)
                    self.commit(E)
                self.model.train_on_batch(X, Y)
                self.iteration += 1
        except StopIteration:
            pass
        # Close the connection with the parameter server.
        self.socket.close()

        return iter([])


class EAMSGDWorker(NetworkWorker):
    """Worker implementation of Asynchronous EA Momentum SGD.

    Introduced by Zhang et al.
    https://arxiv.org/pdf/1412.6651.pdf
    """

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
        self.socket = None
        self.center_variable = None

    def connect(self):
        """Connects with the remote parameter server."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.master_host, self.master_port))

    def pull(self):
        """Fetches the center variable from the parameter server."""
        # Request a pull from the parameter server.
        self.socket.sendall(b'p')
        # Fetch the central variable from the parameter server.
        center_variable = recv_data(self.socket)
        self.center_variable = np.asarray(center_variable)

    def commit(self, delta):
        """Sends the delta to the parameter server."""
        # Prepare the datastructure.
        data = {}
        data['worker_id'] = self.get_worker_id()
        data['delta'] = delta
        # Request a commit from the parameter server.
        self.socket.sendall(b'c')
        # Send the data to the parameter server.
        send_data(self.socket, data)

    def train(self, worker_id, iterator):
        """Specific training procedure of asynchronous EAMSGD."""
        # Prepare the model.
        self.prepare_model()
        # Set the worker identifier.
        self.set_worker_id(worker_id)
        # Connect to the parameter server.
        self.connect()
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
                    self.pull()
                    W = np.asarray(self.model.get_weights())
                    E = self.alpha * (W - self.center_variable)
                    W = W - E
                    self.model.set_weights(W)
                    self.commit(E)
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
        # Close the connection with the parameter server.
        self.socket.close()

        return iter([])
