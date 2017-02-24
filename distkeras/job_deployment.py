"""Module which facilitates job deployment on remote Spark clusters.
This allows you to build models and architectures on, for example, remote
notebook servers, and submit the large scale training job on remote
Hadoop / Spark clusters."""

## BEGIN Imports. ##############################################################

import os

import subprocess

from distkeras.utils import serialize_keras_model

## END Imports. ################################################################

class Job(object):
    """TODO Add documentation"""

    PARAMETER_DISTRIBUTED_OPTIMIZER = "distributed_optimizer"
    PARAMETER_FEATURES_COLUMN = "features_column"
    PARAMETER_KERAS_MODEL = "keras_model"
    PARAMETER_LABEL_COLUMN = "label_column"
    PARAMETER_LOSS = "loss"
    PARAMETER_MINI_BATCH_SIZE = "mini_batch_size"
    PARAMETER_NUM_EPOCH = "num_epoch"
    PARAMETER_NUM_WORKERS = "num_workers"
    PARAMETER_TRAINING_FRACTION = "training_fraction"
    PARAMETER_WORKER_OPTIMIZER = "worker_optimizer"

    def __init__(self, host, username, password):
        self.host = host
        self.username = username
        self.password = password
        self.parameters = {}
        self.initialize_default_parameters()

    def initialize_default_parameters(self):
        raise NotImplementedError

    def get_host(self):
        return self.host

    def get_username(self):
        return self.username

    def get_password(self):
        return self.password

    def generate_code(self):
        raise NotImplementedError

    def copy_code(self):
        raise NotImplementedError

    def copy_result(self):
        raise NotImplementedError

    def process_result(self):
        raise NotImplementedError

    def run(self):
        self.generate_code()
        self.copy_code()
        self.copy_result()

        return self.process_result()
