"""
Module which describes the properties and actions of training a Keras model
on Apache Spark.
"""

## BEGIN Imports. ##############################################################

from distkeras.networking import *
from distkeras.optimizers import *

from flask import Flask, request

from multiprocessing import Process, Lock

import cPickle as pickle

import numpy as np

## END Imports. ################################################################

## BEGIN Utility functions. ####################################################

## END Utility functions. ######################################################

class DistributedModel:

    def __init__(self, keras_model, data, optimizer, master_port=5000):
        self.master_model = keras_model
        self.weights = self.master_model.get_weights()
        self.master_address = determine_host_address()
        self.master_port = master_port
        self.optimizer = optimizer
        self.data = data
        self.mutex = Lock()

    ## BEGIN Flask application. ################################################

    def service(self):
        app = Flask(__name__)
        self.app = all

        ## BEGIN Application routes. ###########################################

        @app.route('/parameters', methods=['GET'])
        def route_parameters():
            with self.mutex:
                pickled_weights = picle.dumps(self.master_model.get_weights())

            return pickled_weights

        @app.route('/update', methods=['POST'])
        def update_parameters():
            delta = pickle.loads(request.data)
            with self.mutex:
                constraints = self.master_model.constraints
                self.weights = self.optimizer.get_updates(self.weights, delta,
                                                          constraints)

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
        return self.master_address + ":" + self.master_port



class SparkModel(DistributedModel):

    def __init__(self, sc, rdd, keras_model, data, optimizer,
                 num_workers=4, master_port=5000):
        # Initialize the super class
        super(SparkModel, self).__init__(keras_model, data,
                                         optimizer, master_port)
        self.spark_context = sc
        self.dataset_rdd = rdd
        self.num_workers = num_workers

    def train(self, parameters):
        self.dataset_rdd = repartition(self.num_workers)
        self._train(self.dataset_rdd, parameters)

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
