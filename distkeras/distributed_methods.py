"""
Distributed learning methods.
"""

## BEGIN Imports. ##############################################################

from flask import Flask, request

from multiprocessing import Process, Lock

import cPickle as pickle

import urllib2

## END Imports. ################################################################

## BEGIN Utility functions. ####################################################

def rest_send_to_master(url, data):
    request = urllib2.Request("http://{0}/update".format(url),
                              pickle.dumps(data, -1),
                              headers={'Content-Type': 'application/dist-keras'})

    return urllib2.urlopen(request).read()

## END Utility functions. ######################################################

class DistributedMethod(object):

    def setup(self):
        raise NotImplementedError

    def run(self, parameters):
        raise NotImplementedError

    def get_master_method(self):
        return self.mater_method

    def get_slave_method(self):
        return self.slave_method

class MasterMethod(object):

    def run(self):
        raise NotImplementedError

class SlaveMethod(object):

    def run(self):
        raise NotImplementedError

class NetworkMasterMethod(MasterMethod):

    def __init__(self, network_port):
        self.network_port = port

    def run(self):
        raise NotImplementedError

class NetworkSlaveMethod(SlaveMethod):

    def __init__(self, master_address, master_port):
        self.master_address = master_address
        self.master_port = master_port

    def get_master_url(self):
        return self.master_address + ":" + `self.master_port`

## BEGIN Synchronous EASGD. ####################################################

class SynchronousEASGDMethod(DistributedMethod):

    def __init__(self, learning_rate, num_workers, rho):
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.rho = rho

    def setup(self):
        # Initialize the master and slave method.
        # TODO Implement.
        self.master_method = None
        self.slave_method = None

    def run(self):
        self.master_method.run()

class SynchronousEASGDMasterMethod(NetworkMasterMethod):

    def __init__(self, network_port, num_workers, rho, num_epoch):
        super(SynchronousEASGDMasterMethod, self).__init__(network_port)
        self.num_workers = num_workers
        self.rho = rho
        self.initialize_server()
        self.num_epoch

    def synchronous_easgd_service(self):
        # Define the master functionality.
        app = Flask()

        ## BEGIN REST routes. ##################################################
        ## END REST routes. ####################################################

        app.run(host='0.0.0.0', port=self.network_port)

    def initialize_server(self):
        service = Process(target=self.synchronous_easgd_service)
        service.start()

    def run(self):
        service.join()

## END Synchronous EASGD. ######################################################
