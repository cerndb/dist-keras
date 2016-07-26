"""
Module which describes the properties and actions of training a Keras model
on Apache Spark.
"""

## BEGIN Imports. ##############################################################

import numpy as np
import cPickle as pickle
from flask import Flask, request
from distkeras.networking import *
from keras.optimizers import *
from multiprocessing import Process, Lock

## END Imports. ################################################################

class SparkModel:

    def __init__(self, sc, keras_model, optimizer, master_port=5000):
        self.spark_context = sc
        self.master_model = keras_model
        self.weights = self.master_model.get_weights()
        self.master_address = determine_host_address()
        self.master_port = master_port
        self.optimizer = optimizer
        self.mutex = Lock()

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

    def start_server(self):
        self.server = Process(target=self.service)
        self.server.start()

    def stop_server(self):
        self.server.terminate()
        self.server.join()
