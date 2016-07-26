"""
DistKeras module which provides some utility functions with
regard to networking.
"""

## BEGIN Imports. ##############################################################

import socket

## END Imports. ################################################################

def determine_host_address():
    host_address = socket.gethostbyname(socket.gethostname())

    return host_address
