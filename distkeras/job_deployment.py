"""Module which facilitates job deployment on remote Spark clusters.
This allows you to build models and architectures on, for example, remote
notebook servers, and submit the large scale training job on remote
Hadoop / Spark clusters."""

## BEGIN Imports. ##############################################################

import os

import subprocess

from distkeras.utils import serialize_keras_model
from distkeras.utils import get_os_username

## END Imports. ################################################################

class Job(object):
    """TODO Add documentation"""

    def __init__(self, job_name, data_path, host, username, password, trainer):
        self.host = host
        self.job_name = job_name
        self.num_executors = 20
        self.num_processes = 1
        self.data_path = data_path
        self.username = username
        self.password = password
        self.trainer = trainer
        self.initialize_default_parameters()
        self.spark_2 = False

    def get_data_path(self):
        return self.data_path

    def set_spark_2(self, using_spark_2):
        self.spark_2 = using_spark_2

    def uses_spark_2(self):
        return self.spark_2

    def set_num_executors(self, num_executors):
        self.num_executors = num_executors

    def set_num_processes(self, num_processes):
        self.num_processes = num_processes

    def num_executors(self):
        return self.num_executors

    def num_processes(self):
        return self.num_processes

    def get_host(self):
        return self.host

    def get_username(self):
        return self.username

    def get_password(self):
        return self.password

    def get_trainer(self):
        return self.trainer

    def generate_code(self):
        # Generate the source code.
        source = """
        # Automatically generated code, do not adapt.
        from distkeras.trainers import *
        from distkeras.predictors import *
        from distkeras.transformers import *
        from distkeras.evaluators import *
        from distkeras.utils import *
        from distkeras.trainers import *
        from keras import *

        import numpy as np
        """
        # Write the source code to a file.
        # TODO Implement.

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
