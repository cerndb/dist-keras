"""Generates a JSON structure that needs to be added to the
secrets file.

Author: Joeri Hermans
"""

## BEGIN Imports. ##############################################################

import json

import optparse

import random

import string

## END Imports. ################################################################

def generate_secret(identity):
    secret = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(64))
    d = {}
    d['secret'] = secret
    d['identity'] = identity
    print(json.dumps(d))

def parse_arguments():
    parser = optparse.OptionParser()
    parser.set_defaults(identity=None)
    parser.add_option('--identity', action='store', dest='identity', type='string')
    (options, args) = parser.parse_args()

    return options

def main():
    # Parse the options.
    options = parse_arguments()
    # Check if an identity has been provided.
    if options.identity is not None:
        generate_secret(options.identity)
    else:
        print("Please specify an identity (--identity).")

if __name__ == '__main__':
    main()
