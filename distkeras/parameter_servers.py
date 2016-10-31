"""Several parameter server implementations.

A parameter server is a process which will aggregate all the incoming gradient
or parameter updates of the workers and incorperate it into a single center variable.
This center variable will eventually be the produced model of the trainer."""

## BEGIN Imports. ##############################################################

from distkeras.networking import *
from distkeras.utils import *

from flask import *

from threading import Lock

import cPickle as pickle

import numpy as np

## END Imports. ################################################################

class ParameterServer(object):

    def __init__(self, model):
        self.model = deserialize_keras_model(model)
        self.num_updates = 0

    def initialize(self):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def get_model(self):
        return self.model

    def next_update(self):
        self.num_updates += 1

    def reset_update_counter(self):
        self.num_updates = 0

    def num_updates(self):
        return self.num_updates

class RESTParameterServer(ParameterServer):

    def __init__(self, model, port):
        super(RESTParameterServer, self).__init__(model)
        self.master_port = port
        self.server = Flask(__name__)

    def start(self):
        pass

    def run(self):

        ## BEGIN Additional REST routes. #######################################

        @self.server.route('/shutdown', methods=['GET'])
        def shutdown():
            f = request.environ.get('werkzeug.server.shutdown')
            f()

            return 'OK'

        ## END Additional REST routes. #########################################

        # Run the REST API server.
        self.server.run(host='0.0.0.0', threaded=True, use_reloader=False)

    def stop(self):
        # Tell the REST server to shutdown.
        rest_get_ping(self.master_host, self.master_port, '/shutdown')

class EAMSGDParameterServer(RESTParameterServer):

    def __init__(self, model, communication_window, rho, learning_rate, momentum, master_port):
        super(EAMSGDParameterServer, self).__init__(model, master_port)
        self.communication_window = communication_window
        self.rho = rho
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.mutex = Lock()

    def initialize(self):

        ## BEGIN EAMSGD REST routes. ###########################################

        @self.server.route('/center_variable', methods=['GET'])
        def center_variable():
            with self.mutex:
                center_variable = self.model.get_weights()

            return pickle.dumps(center_variable, -1)

        @self.server.route('/update', methods=['POST'])
        def update():
            data = pickle.loads(request.data)
            variable = data['variable']
            worker_id = data['worker_id']

            with self.mutex:
                center_variable = self.model.get_weights()
                center_variable = center_variable + variable
                self.model.set_weights(center_variable)
                self.next_update()

            return 'OK'

        ## END EAMSGD REST routes. #############################################
