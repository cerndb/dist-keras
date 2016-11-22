"""
DistKeras module which provides some utility functions for networking.
"""

## BEGIN Imports. ##############################################################

from distkeras.utils import *

import cPickle as pickle

import socket

import urllib2

import zlib

## END Imports. ################################################################

## BEGIN Networking Utility Functions. #########################################

def determine_host_address():
    host_address = socket.gethostbyname(socket.gethostname())

    return host_address

def rest_post(host, port, endpoint, data):
    data = pickle.dumps(data, -1)
    request = urllib2.Request("http://" + host + ":" + `port` + endpoint,
                              data, headers={'Content-Type': 'application/dist-keras'})

    return urllib2.urlopen(request).read()

def rest_get(host, port, endpoint):
    request = urllib2.Request("http://" + host + ":" + `port` + endpoint,
                              headers={'Content-Type': 'application/dist-keras'})

    return pickle.loads(urllib2.urlopen(request).read())

def rest_post_compress(host, port, endpoint, data):
    data = pickle.dumps(data, -1)
    data = compress(data)
    request = urllib2.Request("http://" + host + ":" + `port` + endpoint,
                              data, headers={'Content-Type': 'application/dist-keras'})

    return urllib2.urlopen(request).read()

def rest_get_decompress(host, port, endpoint):
    request = urllib2.Request("http://" + host + ":" + `port` + endpoint,
                              headers={'Content-Type': 'application/dist-keras'})
    data = decompress(urllib2.urlopen(request).read())
    data = pickle.loads(data)

    return data

def rest_get_ping(host, port, endpoint):
    request = urllib2.Request("http://" + host + ":" + `port` + endpoint,
                              headers={'Content-Type': 'application/dist-keras'})
    urllib2.urlopen(request)

## END Networking Utility Functions. ###########################################
