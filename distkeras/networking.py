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

def recvall(socket, n):
    buffer = ''
    buffer_size = 0
    bytes_left = n
    # Iterate until we received all data.
    while buffer_size < n:
        # Fetch the next frame from the network.
        data = socket.recv(bytes_left)
        # Compute the size of the frame.
        delta = len(data)
        buffer_size += delta
        bytes_left -= delta
        # Append the data to the buffer.
        buffer += data

    return buffer

def recv_data(socket):
    data = ''
    # Fetch the serialized data length.
    length = int(recvall(socket, 20).decode())
    # Fetch the serialized data.
    serialized_data = recvall(socket, length)
    # Deserialize the data.
    data = pickle.loads(serialized_data)

    return data

def send_data(socket, data):
    # Serialize the data.
    serialized_data = pickle.dumps(data, -1)
    length = len(serialized_data)
    # Serialize the number of bytes in the data.
    serialized_length = str(length).zfill(20)
    # Send the data over the provided socket.
    socket.sendall(serialized_length.encode())
    socket.sendall(serialized_data)

## END Networking Utility Functions. ###########################################
