"""
Distributed module. This module will contain all distributed classes and
methods.
"""

## BEGIN Imports. ##############################################################

from distkeras.networking import *
from distkeras.parameter_servers import *
from distkeras.utils import *
from distkeras.workers import *

import numpy as np

import threading

import time

## END Imports. ################################################################

class Trainer(object):

    def __init__(self, keras_model, loss, worker_optimizer):
        self.master_model = serialize_keras_model(keras_model)
        self.loss = loss
        self.worker_optimizer = worker_optimizer
        self.history = []
        self.training_time_start = 0
        self.training_time_end = 0
        self.training_time = 0

    def record_training_start(self):
        self.training_time = 0
        self.training_time_start = time.time()

    def record_training_end(self):
        self.training_time_end = time.time()
        self.training_time = self.training_time_start - self.training_time_end

    def get_training_time(self):
        return self.training_time

    def get_history(self):
        return self.history

    def set_history(self, history):
        self.history = history

    def has_history(self):
        return len(self.history) > 0

    def add_history(self, history):
        self.history.append(history)

    def train(self, dataframe, shuffle=False):
        raise NotImplementedError

class SingleTrainer(Trainer):

    def __init__(self, keras_model, worker_optimizer, loss, features_col="features",
                 label_col="label", num_epoch=1, batch_size=32):
        super(SingleTrainer, self).__init__(keras_model, loss, worker_optimizer)
        self.features_column = features_col
        self.label_column = label_col
        self.num_epoch = num_epoch
        self.batch_size = batch_size

    def allocate_worker(self):
        worker = SingleTrainerWorker(keras_model=self.master_model, features_col=self.features_column,
                                     label_col=self.label_column, batch_size=self.batch_size,
                                     optimizer=self.worker_optimizer, loss=self.loss)

        return worker

    def train(self, dataframe, shuffle=False):
        # Check if the data needs to be shuffled.
        if shuffle:
            dataframe = shuffle(dataframe)
        # Collect all the data on a single worker node.
        dataframe = dataframe.coalesce(1)
        # Start recording training time.
        self.record_training_start()
        # Iterate through the number of records.
        for i in range(0, self.num_epoch):
            # Allocate a worker.
            worker = self.allocate_worker()
            # Fetch the trained model.
            self.master_model = dataframe.rdd.mapPartitionsWithIndex(worker.train).collect()[0]
        # Stop recording of training time.
        self.record_training_end()

        return deserialize_keras_model(self.master_model)

class DistributedTrainer(Trainer):

    def __init__(self, keras_model, worker_optimizer, loss, num_workers=2, batch_size=32,
                 features_col="features", label_col="label", num_epoch=1):
        super(DistributedTrainer, self).__init__(self, keras_model, loss, worker_optimizer)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.features_column = features_col
        self.label_column = label_col
        self.num_epoch = num_epoch
        self.parameter_server = None
        self.parameter_server_thread = None

    def allocate_worker(self):
        raise NotImplementedError

    def allocate_parameter_server(self):
        raise NotImplementedError

    def num_updates(self):
        return self.parameter_server.num_updates()

    def service(self):
        self.parameter_server.initialize()
        self.parameter_server.run()

    def stop_service(self):
        self.parameter_server.stop()
        self.parameter_server_thread.join()

    def start_service(self):
        self.parameter_server_thread = threading.Thread(target=self.service)
        self.parameter_server_thread.start()

    def train(self, dataframe, shuffle=False):
        # Start the communication service.
        self.start_service()
        # Allocate a worker.
        worker = self.allocate_worker()
        # Repartition in order to fit the number of workers.
        num_partitions = dataframe.rdd.getNumPartitions()
        # Check if the dataframe needs to be shuffled before training.
        if shuffle:
            dataframe = shuffle(dataframe)
        # Check if we need to repartition the dataframe.
        if num_partitions > self.num_workers:
            dataframe = dataframe.coalesce(self.num_workers)
        else:
            dataframe = dataframe.repartition(self.num_workers)
        # Start the training procedure.
        self.record_training_start()
        # Iterate through the epochs.
        for i in range(0, self.num_epoch):
            dataframe.rdd.mapPartitions(worker.train).collect()
        # End the training procedure.
        self.record_training_end()
        # Stop the communication service.
        self.stop_service

        return self.parameter_server.get_model()

class EAMSGD(DistributedTrainer):

    def __init__(self, keras_model, worker_optimizer, loss, num_workers=2, batch_size=32,
                 features_col="features", label_col="label", communication_window=10,
                 rho=5.0, learning_rate=0.01, momentum=0.9):
        super(EAMSGD, self).__init__(keras_model, worker_optimizer, loss, num_workers,
                                     batch_size, features_col, label_col, num_epoch)
        self.communication_window = communication_window
        self.rho = rho
        self.learning_rate = learning_rate
        self.momentum = momentum

    def allocate_parameter_server(self):
        # Allocate the asynchronous EAMSGD parameter server.
        ps = EAMSGDParameterServer(self.master_model, self.communication_window,
                                   self.rho, self.learning_rate, self.momentum)

        return ps

    def allocate_worker(self):
        raise NotImplementedError
