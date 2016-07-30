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

    def __init__(self, network_port, learning_rate, num_workers, rho, num_epoch):
        self.learning_rate = learning_rate
        self.network_port = network_port
        self.num_workers = num_workers
        self.rho = rho
        self.num_epoch = num_epoch

    def setup(self):
        # Initialize the master and slave method.
        # TODO Implement.
        self.master_method = None
        self.slave_method = None

    def run(self):
        self.master_method.run()

class SynchronousEASGDMasterMethod(NetworkMasterMethod):

    def __init__(self, network_port, num_workers, learning_rate, rho, num_epoch, model):
        super(SynchronousEASGDMasterMethod, self).__init__(network_port)
        self.num_workers = num_workers
        self.rho = rho
        self.initialize_server()
        self.num_epoch = num_epoch
        self.current_epoch = 0
        self.learning_rate = learning_rate
        self.master_mutex = Lock()
        self.master_model = model
        self.epoch_done = True
        self.epoch_weights = {}

    def update_center_variable(self):
        # Fetch the current center variable.
        center_variable = self.master_model.get_weights()
        for x in self.epoch_weights:
            delta += self.rho * (x - center_variable)
        delta *= self.learning_rate
        center_variable += delta
        self.master_model.set_weights(center_variable)
        # Clear the epoch weights dictionary.
        self.epoch_weights.clear()
        # Update the variables for the next epoch.
        self.epoch_done = True
        self.current_epoch += 1

    def synchronous_easgd_service(self):
        # Define the master functionality.
        app = Flask(__name__)

        ## BEGIN REST routes. ##################################################

        @app.route("/center_variable", methods=['GET'])
        def get_variable():
            with self.master_mutex:
                center_variable = self.master_model.get_weights().copy()
                data = {}
                data['current_epoch'] = self.current_epoch
                data['serialized'] = center_variable

            return pickle.dump(data, -1)

        @app.route("/ready", methods=['GET'])
        def epoch_done():
            with self.master_mutex:
                data = pickle.load(request.data)
                epoch = data['epoch']
                if epoch < self.current_epoch:
                    result = "1"
                elif self.epoch_done:
                    result = "1"
                else:
                    result = "0"

            return result

        @app.route("/weights", methods=['POST'])
        def post_weights():
            data = pickle.load(request.data)
            weights = data['weights']
            epoch = data['epoch']
            worker_id = data['worker_id']
            if epoch == current_epoch:
                with self.master_mutex:
                    self.epoch_done = False
                    self.epoch_weights[worker_id] = weights
                    # Check if all workers send their weights
                    if self.epoch_weights.size() == self.num_workers:
                        self.update_center_variable()

        @app.route("/weights", method=['POST'])
        def get_weights():
            with self.master_mutex:
                weights = self.master_model.get_weights()
                serialized = pickle.dump(weights, -1)

            return serialized


        ## END REST routes. ####################################################

        app.run(host='0.0.0.0', threaded=True, use_reloader=False)

    def initialize_server(self):
        service = Process(target=self.synchronous_easgd_service)
        service.start()

    def run(self):
        service.join()


class SynchronousEASGDSlaveMethod(NetworkSlaveMethod):

    def __init__(self, master_address, master_port, learning_rate, rho, num_epoch, json_model):
        super(SynchronousEASGDSlaveMethod, self).__init__(master_address, master_port)
        self.learning_rate = learning_rate
        self.rho = rho
        self.num_epoch = num_epoch
        self.json_model = json_model
        self.model = None

    def prepare_model(self):
        # TODO Implement.
        raise NotImplementedError

    def process_epoch(self, e):
        # TODO Implement.
        raise NotImplementedError

    def run(self):
        self.prepare_model()
        for e in range(0, self.num_epoch):
            self.process_epoch(e)

## END Synchronous EASGD. ######################################################
