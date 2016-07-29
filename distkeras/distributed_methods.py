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

    def __init__(self, master_method, slave_method):
        self.master_method = master_method
        self.slave_method = slave_method

    def get_master_model():
        return self.master_method.master_model

class MasterMethod(object):

    def __init__(self, model):
        self.master_model = model

    def run():
        raise NotImplementedError

class SlaveMethod(object):

    def run():
        raise NotImplementedError

class NetworkMasterMethod(MasterMethod):

    def __init__(self, model, network_port):
        super(MasterMethod, self).__init__(model)
        self.network_port = port

    def run():
        raise NotImplementedError

class NetworkSlaveMethod(SlaveMethod):

    def __init__(self, master_address, master_port):
        self.master_address = master_address
        self.master_port = master_port

    def get_master_url(self):
        return self.master_address + ":" + `self.master_port`

## BEGIN Implementations. ######################################################

class NaiveSGDMasterMethod(NetworkMasterMethod):

    def __init__(self, port, model):
        super(NetworkMasterMethod, self).__init__(model, port)
        self.mutex_master_model = Lock()
        self.server = None

    def run(self):
        self.server = Process(target=self.service)
        self.server.start()

    def service(self):
        raise NotImplementedError

class NaiveSGDSlaveMethod(SlaveMethod):

    def __init__(self, master_address, master_port):
        super(SlaveMethod, self).__init__(master_address, master_port)

    def run(self):
        raise NotImplementedError

## END Implementations . #######################################################
