"""Parameter servers.

A parameter server is a process which will aggregate all the incoming gradient
or parameter updates of the workers and incorperate it into a single center variable.
This center variable will eventually be the produced model of the trainer.
"""

## BEGIN Imports. ##############################################################

import numpy as np

import math

import socket

import threading

from distkeras.networking import recv_data
from distkeras.networking import send_data
from distkeras.utils import deserialize_keras_model

## END Imports. ################################################################

class ParameterServer(object):
    """Abstract class which provides basic attributed and methods for all
       parameter servers.

    # Arguments
        model: string. Serialized Keras model.
               See: distkeras.utils.serialize_keras_model
    """

    def __init__(self, model):
        self.model = deserialize_keras_model(model)
        self.num_updates = 1

    def initialize(self):
        """Initializes the parameter server.

        This method is called after self.start().
        """
        raise NotImplementedError

    def start(self):
        """Starts the parameter server in a new thread."""
        raise NotImplementedError

    def run(self):
        """Main event loop of the parameter server."""
        raise NotImplementedError

    def stop(self):
        """Notifies the parameter server thread to stop."""
        raise NotImplementedError

    def get_model(self):
        """Returns the Keras model which will be trained by the workers."""
        return self.model

    def next_update(self):
        """Increments the number of model updates by 1."""
        self.num_updates += 1

    def reset_update_counter(self):
        """Resets the model update counter."""
        self.num_updates = 0

    def get_num_updates(self):
        """Returns the number of model updates the parameter server has performed."""
        return self.num_updates


class SocketParameterServer(ParameterServer):
    """Abstract class of a parameter server which is based on a socket implementation.

    This means that this parameter server accepts multiple TCP connections from multiple
    workers, and uses a costum protocol to transmit and receive the model parameters. This
    is done by implementing a custom protocol. Which is fully described in the
    distkeras.networking module.

    # Arguments
        model: string. Serialized Keras model.
               See: distkeras.utils.serialize_keras_model
        port: int. Listing port number.
    """

    def __init__(self, model, port):
        super(SocketParameterServer, self).__init__(model)
        self.master_port = port
        self.socket = None
        self.running = False
        self.connections = []
        self.mutex = threading.Lock()

    def initialize(self):
        """Sets up the listing port."""
        # Reset the running flag.
        self.running = True
        # Prepare a socket.
        file_descriptor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Disable Nagle's algorithm.
        file_descriptor.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        file_descriptor.bind(('0.0.0.0', self.master_port))
        # Listen to the socket.
        file_descriptor.listen(5)
        # Assign the socket.
        self.socket = file_descriptor

    def handle_commit(self, conn, addr):
        """Handles parameter updates coming from the workers.

        # Arguments:
            conn: socket. The opened connection.
            addr: addr. Address of the remote host.
        """
        raise NotImplementedError

    def handle_pull(self, conn, addr):
        """Handles parameter requests coming from the workers. This will
        actually send the model parameters to the requesting host.

        # Arguments:
            conn: socket. The opened connection.
            addr: addr. Address of the remote host.
        """
        # Fetch the raw center variables.
        with self.mutex:
            center_variable = self.model.get_weights()
        # Send the data over the socket.
        send_data(conn, center_variable)

    def cancel_accept(self):
        """This method will cancel the accept procedure. The method
        is meant to be executed by the stop() procedure.
        """
        file_descriptor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # Connect to the listening socket to cancel the accept.
            file_descriptor.connect(("localhost", self.master_port))
            file_descriptor.close()
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
        """Starts the parameter server."""
        # Set the running flag.
        self.running = True

    def run(self):
        """Main event loop of the parameter server."""
        # Listen for incoming connections.
        while self.running:
            try:
                # Accept incoming connections.
                conn, addr = self.socket.accept()
                # Handle the connection.
                thread = threading.Thread(target=self.handle_connection, args=(conn, addr))
                thread.start()
                # Store the connection in the dictionary.
                self.connections.append(thread)
            except Exception:
                pass

    def stop(self):
        """Stop the parameter server. This will also cleanup all existing connections."""
        self.running = False
        # Check if a socket is allocated.
        if self.socket:
            self.cleanup_connections()
            self.socket.close()
            self.cancel_accept()
            self.socket = None
        self.connections = []

    def cleanup_connections(self):
        """Clean all existing connections up."""
        # Iterate over all connections.
        for thread in self.connections:
            # Fetch the thread object.
            thread.join()
            del thread


class DeltaParameterServer(SocketParameterServer):
    """A parameter server which integrates all incoming deltas into the model.

    # Arguments
        model: string. Serialized Keras model.
               See: distkeras.utils.serialize_keras_model
        master_port: int. Port number of the parameter server.
    """

    def __init__(self, model, master_port):
        super(DeltaParameterServer, self).__init__(model, master_port)

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


class ADAGParameterServer(SocketParameterServer):
    """A parameter server which integrates the incoming gradient residuals into
       the model, and integrates them using the ADAG scheme.

    # Arguments
        model: string. Keras model.
               See: distkeras.utils.serialize_keras_model
        master_port: int. Port number of the parameter server.
        beta_1: float. Beta_1 in the ADAG algorithm.
        beta_2: float. Beta_2 in the ADAG algorithm.
    """

    def __init__(self, model, master_port):
        super(ADAGParameterServer, self).__init__(model, master_port)

    def handle_commit(self, conn, addr):
        # Receive the parameters from the remote node.
        data = recv_data(conn)
        # Extract the data from the dictionary.
        r = data['residual']
        with self.mutex:
            # Update the center variable.
            center_variable = self.model.get_weights()
            center_variable = center_variable + r
            # Set the new parameters of the model.
            self.model.set_weights(center_variable)
        # Increment the number of parameter server updates.
        self.next_update()
