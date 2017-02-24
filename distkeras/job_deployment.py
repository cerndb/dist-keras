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

    def generate_model(self):
        raise NotImplementedError

    def generate_code(self):
        # Generate the source code.
        source = """
        # Automatically generated code, do not adapt.
        from distkeras.evaluators import *
        from distkeras.predictors import *
        from distkeras.trainers import *
        from distkeras.trainers import *
        from distkeras.transformers import *
        from distkeras.utils import *

        from keras import *

        from pyspark import SparkConf
        from pyspark import SparkContext

        import numpy as np

        # Define the script variables.
        application_name = {application_name}
        num_executors = {num_executors}
        num_processes = {num_processes}
        path_data = {path_data}
        using_spark_2 = {using_spark_2}
        num_workers = num_processes * num_executors

        # Allocate a Spark Context, and a Spark SQL context.
        conf = SparkConf()
        conf.set("spark.app.name", application_name)
        conf.set("spark.master", "yarn-client")
        conf.set("spark.executor.cores", num_processes)
        conf.set("spark.executor.instances", num_executors)
        conf.set("spark.executor.memory", "5g")
        conf.set("spark.locality.wait", "0")
        conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");

        # Read the dataset from HDFS. For now we assume Parquet files.
        if using_spark_2:
            sc = SparkSession.builder.config(conf=conf) \
                             .appName(application_name) \
                              .getOrCreate()
            reader = sc
        else:
            sc = SparkContext(conf=conf)
            from pyspark import SQLContext
            sqlContext = SQLContext(sc)
            reader = sqlContext

        # Read the Parquet datafile, and precache the data on the nodes.
        raw_data = reader.reader.parquet(path_data)
        dataset = precache(raw_data, num_workers)
        """.format(
            application_name=self.job_name,
            num_executors=self.num_executors,
            num_processes=self.num_processes,
            path_data=self.data_path,
            using_spark_2=self.spark_2
        )
        # Write the source code to a file.
        with open(self.username + "-dist-keras-job.py", "w") as f:
            f.write(source)

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
