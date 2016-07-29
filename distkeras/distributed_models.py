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

    def __init__(self, master_method, slave_method):
        self.master_method = master_method
        self.slave_method = slave_method
        self.master_thread = None
        self.slave_thread = None

    def train(self):
        raise NotImplementedError
