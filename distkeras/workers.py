"""Workers module.

This module contains all worker specific implementations for different optimization
algorithms.
"""

## BEGIN Imports. ##############################################################

from distkeras.networking import connect
from distkeras.networking import recv_data
from distkeras.networking import send_data

from distkeras.utils import deserialize_keras_model
from distkeras.utils import serialize_keras_model
from distkeras.utils import set_keras_base_directory
from distkeras.utils import shuffle
from distkeras.utils import uniform_weights

from itertools import tee

from multiprocessing import Pool

import numpy as np

import threading

import sys

# "queue" module in python 3 is named "Queue" in python 2
use_python3 = sys.version_info[0] == 3
if use_python3:
    import queue
else:
    import Queue as queue

import random

import socket

import time

## END Imports. ################################################################

class Worker(object):
    """Abstract class of a worker.

    This class provides basic functionality and properties all workers share.
    """

    def __init__(self, model, optimizer, loss, metrics=["accuracy"], features_col="features", label_col="label",
                 batch_size=32, learning_rate=1.0):
        assert isinstance(features_col, (str, list)), "'features_col' must be a string or a list of strings"
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.metrics= metrics
        self.features_column = [features_col] if isinstance(features_col, str) else features_col
        self.label_column = label_col
        self.batch_size = batch_size
        self.max_mini_batches = 100
        self.prefetching_thread = None
        self.mini_batches = None
        self.is_prefetching = True
        self.worker_id = -1
        self.learning_rate = learning_rate
        self.num_inputs = len(self.features_column)

    def set_max_prefetch(self, max_mini_batches):
        """Sets the maximum number of mini-batches that can be prefetched."""
        self.max_mini_batches = max_mini_batches

    def set_learning_rate(self, learning_rate):
        """Sets the learning rate of the worker."""
        self.learning_rate = learning_rate

    def get_learning_rate(self):
        """Returns the learning rate of the worker."""
        return self.learning_rate

    def set_worker_id(self, worker_id):
        """Sets the worker id.

        # Arguments
            worker_id: int. Worker identifier.
        """
        self.worker_id = worker_id

    def get_worker_id(self):
        """Returns the worker id."""
        return self.worker_id

    def prepare_model(self):
        """Prepares the model for training."""
        # Set the Keras directory.
        set_keras_base_directory()
        # Deserialize the Keras model.
        self.model = deserialize_keras_model(self.model)
        # Compile the model with the specified loss and optimizer.
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

    def get_next_minibatch(self):
        """Returns the next mini-batch."""
        return self.mini_batches.get(timeout=1)

    def start_prefetching_thread(self, iterator):
        """Starts the data prefetching thread."""
        self.mini_batches = queue.Queue()
        self.iterator = iterator
        self.prefetching_thread = threading.Thread(target=self.prefetching)
        self.prefetching_thread.start()

    def prefetching(self):
        try:
            while self.is_prefetching:
                if self.mini_batches.qsize() < self.max_mini_batches:
                    batch = [next(self.iterator) for _ in range(self.batch_size)]
                    iterator_copies = tee(batch, self.num_inputs + 1)
                    feature_iterators = iterator_copies[:-1]
                    label_iterator = iterator_copies[-1]
                    X = [np.asarray([x[self.features_column[i]] for x in iterator]) 
                        for i, iterator in enumerate(feature_iterators)]
                    Y = np.asarray([x[self.label_column] for x in label_iterator])
                    self.mini_batches.put([X, Y])
        except Exception as e:
            print(e)
            self.is_prefetching = False

    def optimize(self):
        """Optimization procedure of a worker."""
        raise NotImplementedError

    def train(self, worker_id, iterator):
        """Training procedure for the worker node.

        # Arguments
            worker_id: int. Partition index provided by Spark. Can be used as a worker_id.
            iterator: iterator. Data iterator.
        """
        # Prepare the optimization procedure.
        self.start_prefetching_thread(iterator)
        self.set_worker_id(worker_id)
        self.prepare_model()
        # Start the optimization procedure.
        try:
            self.optimize()
        except Exception as e:
            # Stop the prefetching process.
            self.is_prefetching = False
            print(e)
        # Wait for the prefetching thread to stop.
        self.prefetching_thread.join()

        return iter([serialize_keras_model(self.model)])


class SequentialWorker(Worker):
    """Implementation for sequential gradient updates on a single worker.

    Will train a model on a single worker node.
    """

    def __init__(self, model, optimizer, loss, metrics=["accuracy"], 
                 features_col="features", label_col="label", batch_size=32):
        # Initialize the parent class.
        super(SequentialWorker, self).__init__(model, optimizer, loss, metrics, features_col,
                                               label_col, batch_size)

    def optimize(self):
        """Training procedure with sequential gradient updates.

        # Returns
            Trained serialized Keras model.
        """
        while True:
            X, Y = self.get_next_minibatch()
            h = self.model.train_on_batch(X, Y)
            self.add_history(h)


class NetworkWorker(Worker):
    """Abstract class of a worker who shares the variables using the network."""

    def __init__(self, model, optimizer, loss, metrics=["accuracy"], features_col="features", label_col="label",
                 batch_size=32, master_host="localhost", master_port=5000, learning_rate=1.0):
        super(NetworkWorker, self).__init__(model, optimizer, loss, metrics, features_col,
                                            label_col, batch_size, learning_rate)
        self.master_host = master_host
        self.master_port = master_port
        self.socket = None
        self.center_variable = None
        self.disable_nagle = True
        self.training_history = []
        self.worker_id = 0

    def connect(self):
        """Connect with the remote parameter server."""
        self.socket = connect(self.master_host, self.master_port, self.disable_nagle)

    def pull(self):
        """Requests the center variable from the parameter server."""
        # Request a pull from the parameter server.
        self.socket.sendall(b'p')
        # Fetch the center variable from the parameter server.
        self.center_variable = np.asarray(recv_data(self.socket))

    def commit(self, residual):
        """Sends the gradient residual to the parameter server."""
        # Prepare the datastructure.
        data = {}
        data['worker_id'] = self.get_worker_id()
        data['delta'] = residual
        # Request a commit from the parameter server.
        self.socket.sendall(b'c')
        # Send the data to the paramter server.
        send_data(self.socket, data)

    def set_tcp_no_delay(self, flag):
        """Disables or enables Nagle's algorithm.
        (True -> TCP_NODELAY = 1)
        (False -> TCP_NODELAY = 0)

        # Arguments:
            flag: boolean. Indicates if Nagle's algorithm should be disabled.
        """
        self.disable_nagle = flag

    def tcp_no_delay(self):
        """Returns the value TCP_NODELAY of the flag (Nagle's algorithm).

        # Returns
            True, if Nagle's algorithm is disabled. False otherwise.
        """
        return self.disable_nagle

    def get_master_host(self):
        """Returns the host address of the master parameter server."""
        return self.master_host

    def get_master_port(self):
        """Returns the port of the master parameter server."""
        return self.master_port

    def add_history(self, h):
        """Appends the specified history data."""
        d = {}
        d['history'] = h
        d['worker_id'] = self.worker_id
        d['iteration'] = self.iteration
        d['timestamp'] = time.time()
        self.training_history.append(d)

    def optimize(self):
        """Optimization procedure of a network worker."""
        raise NotImplementedError

    def train(self, worker_id, iterator):
        """Training procedure of a networked worker with a parameter server."""
        self.start_prefetching_thread(iterator)
        self.set_worker_id(worker_id)
        self.prepare_model()
        self.connect()
        self.pull()
        self.model.set_weights(self.center_variable)
        try:
            self.optimize()
        except Exception as e:
            # Stop the prefetching process.
            self.is_prefetching = False
            print(e)
        self.socket.close()
        self.prefetching_thread.join()

        return iter(self.training_history)


class ADAGWorker(NetworkWorker):
    """Implements the training procedure for ADAG.

    Introduced by Hermans et al.
    """

    def __init__(self, model, optimizer, loss, metrics=["accuracy"], features_col="features", label_col="label",
                 batch_size=32, master_host="localhost", master_port=5000, communication_window=5):
        # Initialize the parent object.
        super(ADAGWorker, self).__init__(model, optimizer, loss, metrics, features_col, label_col,
                                         batch_size, master_host, master_port)
        # Initialize ADAG parameters.
        self.communication_window = communication_window
        self.iteration = 1

    def commit(self, residual):
        """Sends the gradient residual to the parameter server."""
        # Prepare the datastructure.
        data = {}
        data['worker_id'] = self.get_worker_id()
        data['residual'] = residual
        # Request a commit from the parameter server.
        self.socket.sendall(b'c')
        # Send the data to the paramter server.
        send_data(self.socket, data)

    def optimize(self):
        """Optimization procedure of ADAG."""
        W1 = np.asarray(self.model.get_weights())
        while True:
            X, Y = self.get_next_minibatch()
            h = self.model.train_on_batch(X, Y)
            self.add_history(h)
            if self.iteration % self.communication_window == 0:
                W2 = np.asarray(self.model.get_weights())
                delta = W2 - W1
                delta /= self.communication_window
                self.commit(delta)
                self.pull()
                self.model.set_weights(self.center_variable)
                W1 = self.center_variable
            self.iteration += 1


class DOWNPOURWorker(NetworkWorker):
    """Implements the training procedure for the distributed DOWNPOUR optimizer.

    Introduced by Dean et al.
    http://static.googleusercontent.com/media/research.google.com/en//archive/large_deep_networks_nips2012.pdf
    """

    def __init__(self, model, optimizer, loss, metrics=["accuracy"], features_col="features", label_col="label",
                 batch_size=32, master_host="localhost", master_port=5000, communication_window=3):
        # Initialize the parent object.
        super(DOWNPOURWorker, self).__init__(model, optimizer, loss, metrics, features_col, label_col,
                                             batch_size, master_host, master_port)
        self.communication_window = communication_window
        self.iteration = 1

    def optimize(self):
        """Specific optimization procedure for DOWNPOUR."""
        W1 = np.asarray(self.model.get_weights())
        while True:
            X, Y = self.get_next_minibatch()
            if self.iteration % self.communication_window == 0:
                W2 = np.asarray(self.model.get_weights())
                delta = W2 - W1
                self.commit(delta)
                self.pull()
                self.model.set_weights(self.center_variable)
                W1 = self.center_variable
            h = self.model.train_on_batch(X, Y)
            self.add_history(h)
            self.iteration += 1


class AEASGDWorker(NetworkWorker):
    """Implementation of asynchronous EASGD worker.

    Introduced by Zhang et al.
    https://arxiv.org/pdf/1412.6651.pdf
    """

    def __init__(self, model, optimizer, loss, metrics=['accuracy'], features_col="features", label_col="label",
                 batch_size=32, master_host="localhost", master_port=5000, rho=5.0,
                 learning_rate=0.01, communication_window=32):
        # Initialize the parent object.
        super(AEASGDWorker, self).__init__(model, optimizer, loss, metrics, features_col, label_col,
                                           batch_size, master_host, master_port)
        # Initialize AEASGD specific variables.
        self.rho = rho
        self.learning_rate = learning_rate
        self.communication_window = communication_window
        self.alpha = self.rho * self.learning_rate
        self.iteration = 1

    def optimize(self):
        """Specific training procedure for AEASGD."""
        while True:
            X, Y = self.get_next_minibatch()
            if self.iteration % self.communication_window == 0:
                self.pull()
                W = np.asarray(self.model.get_weights())
                E = self.alpha * (W - self.center_variable)
                W = W - E
                self.model.set_weights(W)
                self.commit(E)
            h = self.model.train_on_batch(X, Y)
            self.add_history(h)
            self.iteration += 1


class EAMSGDWorker(NetworkWorker):
    """Worker implementation of Asynchronous EA Momentum SGD.

    Introduced by Zhang et al.
    https://arxiv.org/pdf/1412.6651.pdf
    """

    def __init__(self, model, optimizer, loss, metrics=['accuracy'], features_col="features", label_col="label",
                 batch_size=32, master_host="localhost", master_port=5000, rho=5.0,
                 learning_rate=0.01, momentum=0.9, communication_window=32):
        # Initialize the parent object.
        super(EAMSGDWorker, self).__init__(model, optimizer, loss, metrics, features_col, label_col,
                                           batch_size, master_host, master_port)
        # Initialize EAMSGD specific variables.
        self.rho = rho
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.communication_window = communication_window
        self.alpha = self.learning_rate * self.rho
        self.iteration = 1

    def optimize(self):
        """Specific training procedure of asynchronous EAMSGD."""
        r = np.asarray(self.model.get_weights())
        r.fill(0.0)
        while True:
            X, Y = self.get_next_minibatch()
            if self.iteration % self.communication_window == 0:
                self.pull()
                W = np.asarray(self.model.get_weights())
                E = self.alpha * (W - self.center_variable)
                W = W - E
                self.model.set_weights(W)
                self.commit(E)
            r_t = self.momentum * r
            W_copy = np.asarray(self.model.get_weights())
            W = np.asarray(self.model.get_weights())
            W += r_t
            self.model.set_weights(W)
            h = self.model.train_on_batch(X, Y)
            self.add_history(h)
            gradient = np.asarray(self.model.get_weights()) - W
            r = r_t - self.learning_rate * gradient
            W_copy -= r
            self.model.set_weights(W_copy)
            self.iteration += 1


class DynSGDWorker(NetworkWorker):
    """Implements the training procedure for DynSGD."""

    def __init__(self, model, optimizer, loss, metrics=["accuracy"], features_col="features", label_col="label",
                 batch_size=32, master_host="localhost", master_port=5000, communication_window=5):
        # Initialize the parent object.
        super(DynSGDWorker, self).__init__(model, optimizer, loss, metrics, features_col, label_col,
                                           batch_size, master_host, master_port)
        # Initialize DynSGD parameters.
        self.communication_window = communication_window
        self.iteration = 1
        self.last_update = 0

    def pull(self):
        """Requests the center variable and last update from the parameter server."""
        # Request a pull from the parameter server.
        self.socket.sendall(b'p')
        # Fetch the dictionary from the parameter server.
        data = recv_data(self.socket)
        self.center_variable = np.asarray(data['model'])
        self.last_update = data['update']

    def commit(self, residual):
        """Sends the gradient residual to the parameter server."""
        # Prepare the datastructure.
        data = {}
        data['worker_id'] = self.get_worker_id()
        data['residual'] = residual
        data['last_update'] = self.last_update
        # Request a commit from the parameter server.
        self.socket.sendall(b'c')
        # Send the data to the paramter server.
        send_data(self.socket, data)

    def optimize(self):
        """Optimization procedure of DynSGD."""
        W1 = np.asarray(self.model.get_weights())
        while True:
            X, Y = self.get_next_minibatch()
            h = self.model.train_on_batch(X, Y)
            self.add_history(h)
            if self.iteration % self.communication_window == 0:
                W2 = np.asarray(self.model.get_weights())
                delta = W2 - W1
                self.commit(delta)
                self.pull()
                self.model.set_weights(self.center_variable)
                W1 = self.center_variable
            self.iteration += 1


class ExperimentalWorker(NetworkWorker):
    """Implements the training procedure for ADAG.

    Introduced by Hermans et al.
    """

    def __init__(self, model, optimizer, loss, metrics=["accuracy"], features_col="features", label_col="label",
                 batch_size=32, master_host="localhost", master_port=5000, communication_window=5,
                 num_workers=2, learning_rate=1.0):
        # Initialize the parent object.
        super(ExperimentalWorker, self).__init__(model, optimizer, loss, metrics, features_col, label_col,
                                                 batch_size, master_host, master_port, learning_rate)
        # Initialize ADAG parameters.
        self.communication_window = communication_window
        self.num_workers = num_workers
        self.current_num_workers = self.num_workers
        self.inverse_learning_rate = 1 / self.learning_rate
        self.iteration = 1

    def commit(self, residual):
        """Sends the gradient residual to the parameter server."""
        # Prepare the datastructure.
        data = {}
        data['worker_id'] = self.get_worker_id()
        data['residual'] = residual
        data['stale_center_variable'] = self.center_variable
        # Request a commit from the parameter server.
        self.socket.sendall(b'c')
        # Send the data to the paramter server.
        send_data(self.socket, data)

    def pull(self):
        """Requests the center variable from the parameter server."""
        # Request a pull from the parameter server.
        self.socket.sendall(b'p')
        # Fetch the center variable from the parameter server.
        self.center_variable = np.asarray(recv_data(self.socket))

    def optimize(self):
        """Optimization procedure of ADAG."""
        W1 = np.asarray(self.model.get_weights())
        while True:
            X, Y = self.get_next_minibatch()
            h = self.model.train_on_batch(X, Y)
            self.add_history(h)
            if self.iteration % self.communication_window == 0:
                W2 = np.asarray(self.model.get_weights())
                delta = W2 - W1
                delta /= self.communication_window
                self.commit(delta)
                self.pull()
                self.model.set_weights(self.center_variable)
                W1 = self.center_variable
            self.iteration += 1
