"""Script which starts the Punchcard daemon. Punchcard will accept remote job
requests and execute them on the local cluster.

Author: Joeri Hermans
"""

## BEGIN Imports. ##############################################################

from distkeras.job_deployment import Job
from distkeras.job_deployment import Punchcard

import os

import sys

import optparse

## END Imports. ################################################################

def parse_arguments():
    parser = optparse.OptionParser()
    parser.set_defaults(port=8000, secrets_path='secrets.json')
    parser.add_option('--port', action='store', dest='port', type='int')
    parser.add_option('--secrets', action='store', dest='secrets_path', type='string')
    (options, args) = parser.parse_args()

    return options

def start_punchcard(port, secrets):
    punchcard = Punchcard(secrets, port)
    punchcard.run()

def main():
    # Parse the program arguments.
    options = parse_arguments()
    port = options.port
    secrets_path = options.secrets_path
    # Start the Punchcard instance.
    start_punchcard(port, secrets_path)

if __name__ == '__main__':
    main()
