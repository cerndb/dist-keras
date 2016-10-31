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
        self.num_updates = 1

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

class DOWNPOURParameterServer(RESTParameterServer):

    def __init__(self, model, learning_rate, master_port):
        super(DOWNPOURParameterServer, self).__init__(model, master_port)
        self.learning_rate = learning_rate
        self.mutex = Lock()

    def initialize(self):

        ## BEGIN DOWNPOUR REST routes. #########################################

        @self.server.route('/center_variable', methods=['GET'])
        def center_variable():
            with self.mutex:
                center_variable = self.model.get_weights()

            return center_variable

        @self.server.route('/update', methods=['POST'])
        def update():
            data = pickle.loads(request.data)
            variable = data['variable']

            with self.mutex:
                center_variable = self.model.get_weights()
                center_variable = center_variable + variable
                self.model.set_weights(center_variable)
                self.next_update()

            return 'OK'

        ## END DOWNPOUR REST routes. ###########################################

class EASGDParameterServer(RESTParameterServer):

    def __init__(self, model, rho, learning_rate, master_port, num_workers):
        super(EASGDParameterServer, self).__init__(model, master_port)
        self.rho = rho
        self.learning_rate = learning_rate
        self.mutex = Lock()
        self.num_workers = num_workers
        self.ready_mutex = Lock()
        self.variables = {}
        self.ready = False

    def is_ready(self):
        with self.ready_mutex:
            ready = self.ready

        return ready

    def set_ready(self, ready):
        with self.ready_mutex:
            self.ready = ready

    def process_variables(self):
        center_variable = self.model.get_weights()
        temp = np.copy(center_variable)
        temp.fill(0.0)

        for i in range(0, self.num_workers):
            temp += (self.rho * (self.variables[i] - center_variable))
        temp /= float(self.num_workers)
        temp *= self.learning_rate
        center_variable += temp
        self.model.set_weights(center_variable)

    def initialize(self):

        ## BEGIN EASGD REST routes. ############################################

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
            iteration = data['iteration']

            self.set_ready(False)
            # Check if the variable update is within the correct iteration.
            if self.num_updates == self.iteration:
                with self.mutex:
                    self.variables[worker_id] = variable
                    num_variables = len(self.variables)
                # Check if all parameter updates are available.
                if num_variables == self.num_workers:
                    self.process_variables()
                    self.variables = {}
                    self.set_ready(True)
                    self.next_update()

            return 'OK'

        @self.server.route('/ready', methods=['POST'])
        def ready():
            data = pickle.loads(request.data)
            iteration = data['iteration']
            ready = self.is_ready()
            ready = (ready or iteration < self.iteration)

            return str(int(ready))

        ## END EASGD REST routes. ##############################################

class AEASGDParameterServer(RESTParameterServer):

    def __init__(self, model, rho, learning_rate, master_port):
        super(AEASGDParameterServer, self).__init__(model, master_port)
        self.rho = rho
        self.learning_rate = learning_rate
        self.mutex = Lock()

    def initialize(self):

        ## BEGIN AEASGD REST routes. ###########################################

        @self.server.route('/center_variable', methods=['GET'])
        def center_variable():
            with self.mutex:
                center_variable = self.model.get_weights()

            return pickle.dumps(center_variable, -1)

        @self.server.route('/update', methods=['POST'])
        def update():
            data = pickle.loads(request.data)
            variable = data['variable']

            with self.mutex:
                center_variable = self.model.get_weights()
                center_variable = center_variable + variable
                self.model.set_weights(center_variable)
                self.next_update()

            return 'OK'

        ## END AEASGD REST routes. #############################################

class EAMSGDParameterServer(RESTParameterServer):

    def __init__(self, model, rho, learning_rate, momentum, master_port):
        super(EAMSGDParameterServer, self).__init__(model, master_port)
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

            with self.mutex:
                center_variable = self.model.get_weights()
                center_variable = center_variable + variable
                self.model.set_weights(center_variable)
                self.next_update()

            return 'OK'

        ## END EAMSGD REST routes. #############################################
