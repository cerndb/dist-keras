"""
DistKeras module which provides some utility functions for networking.
"""

import socket

def determine_host_address():
    host_address = socket.gethostbyname(socket.gethostname())

    return host_address
