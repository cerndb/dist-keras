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

    def run(self):
        raise NotImplementedError

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
