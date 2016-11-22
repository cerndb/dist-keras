"""
Several parameter server implementations.

A parameter server is a process which will aggregate all the incoming gradient
or parameter updates of the workers and incorperate it into a single center variable.
This center variable will eventually be the produced model of the trainer.
"""

## BEGIN Imports. ##############################################################

from distkeras.networking import *
from distkeras.utils import *

from threading import Lock

from flask import *

import cPickle as pickle

import numpy as np

import threading

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
        rest_get_ping('localhost', self.master_port, '/shutdown')

class SocketParameterServer(ParameterServer):

    def __init__(self, model, port):
        super(SocketParameterServer, self).__init__(model)
        self.master_port = port
        self.socket = None
        self.running = False
        self.connections = []

    def initialize(self):
        # Prepare a socket.
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Disable Nagle's algorithm.
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.bind(('0.0.0.0', self.master_port))
        # Listen to the socket.
        s.listen(5)
        # Assign the socket.
        self.socket = s

    def handle_commit(self, conn, addr):
        raise NotImplementedError

    def handle_pull(self, conn, addr):
        # Fetch the raw center variables.
        with self.mutex:
            center_variable = self.model.get_weights()
        # Send the data over the socket.
        send_data(conn, center_variable)

    def handle_connection(self, conn, addr):
        """
        A parameter server has two main functionalities. Nodes are able to
        pull (p) the current state, or 'commit' a state. This is implemented
        in the following functionality. Classes which implement these interfaces
        should not worry about connection handling.
        """
        while self.running:
            # Fetch the current action.
            action = conn.recv(1).decode()
            # Check if the action is a commit (most of the cases).
            if action == 'c':
                # Handle the commit.
                self.handle_commit(conn, addr)
            elif action == 'p':
                # Handle the pull.
                self.handle_pull(conn, addr)

    def start(self):
        # Set the running flag.
        self.running = True

    def run(self):
        # Listen for incoming connections.
        while self.running:
            # Accept incoming connections.
            conn, addr = self.socket.accept()
            # Handle the connection.
            t = threading.Thread(target=self.handle_connection, args=(conn, addr))
            t.start()
            # Store the connection in the dictionary.
            self.connections.append(t)

    def stop(self):
        self.running = False
        # Check if a socket is allocated.
        if self.socket:
            self.socket.close()
            self.cleanup_connections()
            self.socket = None
        self.connections = []

    def cleanup_connections(self):
        # Iterate over all connections.
        for t in self.connections:
            # Fetch the thread object.
            t.join()
            del t

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

        ## END DOWNPOUR REST routes. ###########################################

class DOWNPOURSocketParameterServer(SocketParameterServer):

    def __init__(self, model, learning_rate, master_port):
        super(DOWNPOURSocketParameterServer, self).__init__(model, master_port)
        self.learning_rate = learning_rate
        self.mutex = Lock()

    def handle_commit(self, conn, addr):
        # Receive the parameters from the remote node.
        data = recv_data(conn)
        # Extract the delta from the dictionary.
        delta = data['delta']

        # Update the center variable with the delta.
        with self.mutex:
            # Fetch the center variable.
            center_variable = self.model.get_weights()
            center_variable = center_variable + delta
            # Set the new parameters of the model.
            self.model.set_weights(center_variable)
            self.next_update()

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
            # Compress the center variable.
            center_variable = pickle.dumps(center_variable, -1)

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
