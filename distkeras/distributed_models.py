"""
Module which describes the properties and actions of training a Keras model
on Apache Spark.
"""

## BEGIN Imports. ##############################################################

from __future__ import absolute_import

from distkeras.networking import *

from keras.models import model_from_json
from keras.engine.training import slice_X

from flask import Flask, request

from multiprocessing import Process, Lock

import cPickle as pickle

from itertools import tee

import numpy as np

import urllib2

## END Imports. ################################################################

## BEGIN Utility functions. ####################################################

def to_simple_rdd(sc, features, labels):
    pairs = [(x, y) for x, y in zip(features, labels)]

    return sc.parallelize(pairs)

def get_master_weights(url):
    request = urllib2.Request("http://{0}/parameters".format(url),
                              headers={'Content-Type': 'application/dist-keras'})

    return pickle.loads(urllib2.urlopen(request).read())

def send_master_deltas(url, deltas):
    request = urllib2.Request("http://{0}/update".format(url),
                              pickle.dumps(deltas, -1),
                              headers={'Content-Type': 'application/dist-keras'})

    return urllib2.urlopen(request).read()

def subtract_params(p1, p2):
    result = []
    for x, y in zip(p1, p2):
        result.append(x - y)

    return result

def add_params(p1, p2):
    result = []
    for x, y in zip(p1, p2):
        result.append(x + y)

    return result

## END Utility functions. ######################################################

class DistributedModel(object):

    def __init__(self, distributed_method):
        self.distributed_method = distributed_method
        self.model_setup = False

    def setup(self):
        self.distributed_method.setup()
        self.model_setup = True

    def train(self):
        raise NotImplementedError

    def is_setup(self):
        return self.model_setup

class SparkModel(DistributedModel):

    def __init__(self, distributed_method, sc, num_workers):
        super(SparkModel, self).__init__(distributed_method)
        self.spark_context = sc
        self.num_workers

    def train(self):
        # Check if the model was setup.
        if not self.is_setup:
            raise ValueError
        # Run the distributed method
        self.distributed_method.run()
