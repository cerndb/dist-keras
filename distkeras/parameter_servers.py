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

class SocketParameterServer(ParameterServer):

    def __init__(self, model, port):
        super(SocketParameterServer, self).__init__(model)
        self.master_port = port
        self.socket = None
        self.running = False
        self.connections = []

    def initialize(self):
        # Reset the running flag.
        self.running = True
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

    def cancel_accept(self):
        """
        This method will cancel the accept procedure. The method
        is meant to be executed by the stop() procedure.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # Connect to the listening socket to cancel the accept.
            s.connect(("localhost", self.master_port))
            s.close()
        except Exception:
            pass

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
            try:
                # Accept incoming connections.
                conn, addr = self.socket.accept()
                # Handle the connection.
                t = threading.Thread(target=self.handle_connection, args=(conn, addr))
                t.start()
                # Store the connection in the dictionary.
                self.connections.append(t)
            except Exception:
                pass

    def stop(self):
        self.running = False
        # Check if a socket is allocated.
        if self.socket:
            self.cleanup_connections()
            self.socket.close()
            self.cancel_accept()
            self.socket = None
        self.connections = []

    def cleanup_connections(self):
        # Iterate over all connections.
        for t in self.connections:
            # Fetch the thread object.
            t.join()
            del t

class DeltaParameterServer(SocketParameterServer):

    def __init__(self, model, master_port):
        super(DeltaParameterServer, self).__init__(model, master_port)
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
        # Next iteration.
        self.next_update()
