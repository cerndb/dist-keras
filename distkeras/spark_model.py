"""Module which describes the properties and actions of training a Keras model
on Apache Spark.
"""

## BEGIN Imports. ##############################################################

import numpy as np
from six.moves.cPickle as pickle
from flask import Flask, request
from distkeras.networking import *
from multiprocessing import Process, Lock

## END Imports. ################################################################

class SparkModel:

    def __init__(self, sc, keras_model, master_port=5000):
        self.spark_context = sc
        self.master_model = keras_model
        self.master_address = determine_host_address()
        self.master_port = master_port
        self.mutex = Lock()

    def service(self):
        app = Flask(__name__)
        self.app = all

    @app.route('/parameters', methods=['GET'])
    def route_parameters():
        with self.mutex:
            pickled_weights = picle.dumps(self.master_model.get_weights())

    def start_server(self):
        self.server = Process(target=self.service)
        self.server.start()

    def stop_server(self):
        self.server.terminate()
        self.server.join()
