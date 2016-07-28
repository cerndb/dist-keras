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

    def __init__(self, keras_model, optimizer, master_port=5000):
        self.master_model = keras_model
        self.weights = self.master_model.get_weights()
        self.master_address = determine_host_address()
        self.master_port = master_port
        self.optimizer = optimizer
        self.mutex = Lock()

    ## BEGIN Flask application. ################################################

    def service(self):
        app = Flask(__name__)
        self.app = app

        ## BEGIN Application routes. ###########################################

        @app.route('/parameters', methods=['GET'])
        def route_parameters():
            with self.mutex:
                pickled_weights = pickle.dumps(self.master_model.get_weights(), -1)

            return pickled_weights

        @app.route('/update', methods=['POST'])
        def update_parameters():
            deltas = pickle.loads(request.data)
            with self.mutex:
                print("\n\n\n\nUpdating weights\n\n\n\n\n")
                weights = self.master_model.get_weights()
                weights += deltas
                self.master_model.set_weights(weights)
            return "OK"

        ## END Application routes. #############################################

        # Run the weights API.
        self.app.run(host='0.0.0.0', threaded=True, use_reloader=False)

    ## END Flask application. ##################################################

    def start_server(self):
        self.server = Process(target=self.service)
        self.server.start()

    def stop_server(self):
        self.server.terminate()
        self.server.join()

    def get_config(self):
        model_config = {}
        model_config['model'] = self.master_model.get_config()
        model_config['optimizer'] = self.optimizer.get_config()
        model_config['mode'] = self.mode

        return model_config

    def predict(self, data):
        return self.master_model.predict(data)

    def predict_classes(self, data):
        return self.master_model.predict_classes(data)

    def train(self, parameters):
        raise NotImplementedError

    def get_master_url(self):
        return self.master_address + ":" + `self.master_port`



class SparkModel(DistributedModel):

    def __init__(self, sc, rdd, keras_model, optimizer, loss,
                 num_workers=1, master_port=5000):
        # Initialize the super class
        super(SparkModel, self).__init__(keras_model, optimizer, master_port)
        self.spark_context = sc
        self.dataset_rdd = rdd
        self.loss = loss
        self.frequency = 'epoch'
        self.num_workers = num_workers

    def train(self, parameters):
        # Start the weights service
        self.start_server()
        self.dataset_rdd = self.dataset_rdd.repartition(self.num_workers)
        self._train(parameters)

    def _train(self, parameters):
        json_model = self.master_model.to_json()
        master_url = self.get_master_url()
        worker = SparkWorker(json_model=json_model,
                             optimizer=self.optimizer,
                             loss=self.loss,
                             train_config=parameters,
                             frequency=self.frequency,
                             master_url=master_url)
        self.dataset_rdd.mapPartitions(worker.train).collect()
        new_weights = get_master_weights(master_url)
        # Check if valid parameters have been received.
        if( len(new_weights) != 0):
            self.master_model.set_weights(new_weights)
        self.stop_server()


class SparkWorker(object):

    def __init__(self, json_model, optimizer, loss, train_config, frequency,
                 master_url):
        self.json_model = json_model
        self.optimizer = optimizer
        self.loss = loss
        self.train_config = train_config
        self.master_url = master_url
        self.frequency = frequency
        self.nb_epoch = train_config['nb_epoch']
        self.batch_size = train_config['batch_size']

    def train(self, data_iterator):
        feature_iterator, label_iterator = tee(data_iterator, 2)
        x_train = np.asarray([x for x, y in feature_iterator])
        y_train = np.asarray([y for x, y in label_iterator])
        # Check if a valid number of features have been provided.
        if( x_train.size == 0 ):
            return

        # Construct a Keras model from the specified JSON string.
        model = model_from_json(self.json_model)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        # Fetch the training parameters from the configuration.
        nb_epoch = self.nb_epoch
        batch_size = self.batch_size
        nb_train_sample = len(x_train[0])
        np_batch = int(np.ceil(nb_train_sample / float(batch_size)))
        index_array = np.arange(nb_train_sample)
        batches = [(i * batch_size, min(nb_train_sample, (i + 1) * batch_size)) for i in range(0, batch_size)]
        if( self.frequency == 'epoch' ):
            for epoch in range(nb_epoch):
                nb_epoch = 1
                batch_size = self.train_config['batch_size']
                # Fetch the weights before the traiing
                weights_before = get_master_weights(self.master_url)
                if( len(weights_before) > 0):
                    model.set_weights(weights_before)
                model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch)
                weights_after = model.get_weights()
                deltas = subtract_params(weights_before, weights_after)
                send_master_deltas(self.master_url, deltas)

        yield []
