"""
Distributed module. This module will contain all distributed classes and
methods.
"""

## BEGIN Imports. ##############################################################

from distkeras.networking import *
from distkeras.utils import *
from distkeras.workers import *

from flask import Flask, request

from pyspark.mllib.linalg import DenseVector

from threading import Lock

import cPickle as pickle

import numpy as np

import threading

## END Imports. ################################################################

## BEGIN Transformers. #########################################################

class Transformer(object):

    def transform(self, data):
        raise NotImplementedError

class LabelVectorTransformer(Transformer):

    def __init__(self, output_dim, input_col="label", output_col="label_vectorized"):
        self.input_column = input_col
        self.output_column = output_col
        self.output_dim = output_dim

    def _transform(self, iterator):
        rows = []
        try:
            for row in iterator:
                label = row[self.input_column]
                v = to_dense_vector(label, self.output_dim)
                new_row = new_dataframe_row(row, self.output_column, v)
                rows.append(new_row)
        except TypeError:
            pass

        return iter(rows)

    def transform(self, data):
        return data.rdd.mapPartitions(self._transform)

class LabelIndexTransformer(Transformer):

    def __init__(self, output_dim, input_col="prediction", output_col="predicted_index",
                 default_index=0, activation_threshold=0.55):
        self.input_column = input_col
        self.output_column = output_col
        self.output_dim = output_dim
        self.activation_threshold = activation_threshold
        self.default_index = default_index

    def get_index(self, vector):
        for index in range(0, self.output_dim):
            if vector[index] >= self.activation_threshold:
                return index
        return self.default_index

    def _transform(self, iterator):
        rows = []
        try:
            for row in iterator:
                output_vector = row[self.input_column]
                index = float(self.get_index(output_vector))
                new_row = new_dataframe_row(row, self.output_column, index)
                rows.append(new_row)
        except ValueError:
            pass

        return iter(rows)

    def transform(self, data):
        return data.rdd.mapPartitions(self._transform)

## END Transformers. ###########################################################

## BEGIN Predictors. ###########################################################

class Predictor(Transformer):

    def __init__(self, keras_model):
        self.model = serialize_keras_model(keras_model)

    def predict(self, data):
        raise NotImplementedError

class ModelPredictor(Predictor):

    def __init__(self, keras_model, features_col="features", output_col="prediction"):
        super(ModelPredictor, self).__init__(keras_model)
        self.features_column = features_col
        self.output_column = output_col

    def _predict(self, iterator):
        rows = []
        model = deserialize_keras_model(self.model)
        try:
            for row in iterator:
                X = np.asarray([row[self.features_column]])
                Y = model.predict(X)
                v = DenseVector(Y[0])
                new_row = new_dataframe_row(row, self.output_column, v)
                rows.append(new_row)
        except ValueError:
            pass

        return iter(rows)

    def predict(self, data):
        return data.rdd.mapPartitions(self._predict)

## END Predictors. #############################################################

## BEGIN Trainers. #############################################################

class Trainer(object):

    def __init__(self, keras_model, loss, worker_optimizer):
        self.master_model = serialize_keras_model(keras_model)
        self.loss = loss
        self.worker_optimizer = worker_optimizer

    def train(self, data, shuffle=False):
        raise NotImplementedError

class SingleTrainer(Trainer):

    def __init__(self, keras_model, worker_optimizer, loss, features_col="features",
                 label_col="label", num_epoch=1, batch_size=32):
        super(SingleTrainer, self).__init__(keras_model, loss, worker_optimizer)
        self.features_column = features_col
        self.label_column = label_col
        self.num_epoch = num_epoch
        self.batch_size = batch_size

    def train(self, data, shuffle=False):
        data = data.coalesce(1)
        if shuffle:
            data = shuffle(data)
        # Fetch the master model.
        model = self.master_model
        for i in range(0, self.num_epoch):
            # Allocate a worker.
            worker = SingleTrainerWorker(keras_model=model, features_col=self.features_column,
                                         label_col=self.label_column, batch_size=self.batch_size,
                                         worker_optimizer=self.worker_optimizer, loss=self.loss)
            # Fetch the trained model.
            model = data.rdd.mapPartitions(worker.train).collect()
        model = deserialize_keras_model(model[0])

        return model

## BEGIN Asynchronous trainers. ################################################

class AsynchronousDistributedTrainer(Trainer):

    def __init__(self, keras_model, worker_optimizer, loss, num_workers=2, batch_size=32,
                 features_col="features", label_col="label", num_epoch=1):
        super(AsynchronousDistributedTrainer, self).__init__(keras_model, loss, worker_optimizer)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.features_column = features_col
        self.label_column = label_col
        self.iteration = 1
        self.parameter_server = None
        self.mutex = Lock()
        self.num_epoch = num_epoch
        self.model = None

    def reset_variables(self):
        self.iteration = 1

    def start_service(self):
        self.parameter_server = threading.Thread(target=self.service)
        self.parameter_server.start()

    def service(self):
        raise NotImplementedError

    def stop_service(self):
        raise NotImplementedError

    def allocate_worker(self):
        raise NotImplementedError

    def train(self, data, shuffle=False):
        # Start the communication service.
        self.start_service()
        # Allocate a worker program.
        worker = self.allocate_worker()
        numPartitions = data.rdd.getNumPartitions()
        if numPartitions > self.num_workers:
            data = data.coalesce(self.num_workers)
        else:
            data = data.repartition(self.num_workers)
        if shuffle:
            data = shuffle(data)
        for i in range(0, self.num_epoch):
            self.reset_variables()
            data.rdd.mapPartitionsWithIndex(worker.train).collect()
        self.stop_service()

        return self.model

class AsynchronousEAMSGD(AsynchronousDistributedTrainer):

    def __init__(self, keras_model, worker_optimizer, loss, num_workers=2, batch_size=32,
                 features_col="features", label_col="label", communication_window=10,
                 rho=5.0, learning_rate=0.01, momentum=0.9, master_port=5000, num_epoch=1):
        super(AsynchronousEAMSGD, self).__init__(keras_model=keras_model, num_workers=num_workers,
                                                 batch_size=batch_size, features_col=features_col,
                                                 label_col=label_col, worker_optimizer=worker_optimizer,
                                                 loss=loss, num_epoch=num_epoch)
        # Initialize the algorithm parameters.
        self.learning_rate = learning_rate
        self.rho = rho
        self.momentum = momentum
        self.communication_window = communication_window
        # Initialize the master server parameters.
        self.master_host = determine_host_address()
        self.master_port = master_port
        # Initialize the default model parameters.
        self.initialize_variables()

    def initialize_variables(self):
        self.model = deserialize_keras_model(self.master_model)

    def stop_service(self):
        rest_get_ping(self.master_host, self.master_port, '/shutdown')
        self.parameter_server.join()

    def allocate_worker(self):
        worker = AsynchronousEAMSGDWorker(keras_model=self.master_model,
                                          features_col=self.features_column,
                                          label_col=self.label_column,
                                          rho=self.rho,
                                          learning_rate=self.learning_rate,
                                          communication_window=self.communication_window,
                                          momentum=self.momentum,
                                          batch_size=self.batch_size,
                                          master_host=self.master_host,
                                          master_port=self.master_port,
                                          worker_optimizer=self.worker_optimizer,
                                          loss=self.loss)

        return worker

    def service(self):
        app = Flask(__name__)

        ## BEGIN REST routes. ##################################################

        @app.route('/center_variable', methods=['GET'])
        def center_variable():
            with self.mutex:
                center_variable = self.model.get_weights()

            return pickle.dumps(center_variable, -1)

        @app.route('/update', methods=['POST'])
        def update():
            data = pickle.loads(request.data)
            variable = data['variable']
            iteration = data['iteration']
            worker_id = data['worker_id']

            with self.mutex:
                center_variable = self.model.get_weights()
                center_variable = center_variable + variable
                self.model.set_weights(center_variable)
                self.iteration += 1

            return 'OK'

        @app.route('/shutdown', methods=['GET'])
        def shutdown():
            f = request.environ.get('werkzeug.server.shutdown')
            f()

            return 'OK'

        ## END REST routes. ####################################################

        app.run(host='0.0.0.0', threaded=True, use_reloader=False)

class AsynchronousEASGD(AsynchronousDistributedTrainer):

    def __init__(self, keras_model, worker_optimizer, loss, num_workers=2, batch_size=32,
                 features_col="features", label_col="label", communication_window=10,
                 rho=5.0, learning_rate=0.01, master_port=5000, num_epoch=1):
        super(AsynchronousEASGD, self).__init__(keras_model=keras_model, num_workers=num_workers,
                                                batch_size=batch_size, features_col=features_col,
                                                label_col=label_col, worker_optimizer=worker_optimizer,
                                                loss=loss, num_epoch=num_epoch)
        # Initialize the algorithm parameters.
        self.learning_rate = learning_rate
        self.rho = rho
        self.communication_window = communication_window
        # Initialize the master server parameters.
        self.master_host = determine_host_address()
        self.master_port = master_port
        # Initialize the default model parameters.
        self.initialize_variables()

    def initialize_variables(self):
        self.model = deserialize_keras_model(self.master_model)

    def stop_service(self):
        rest_get_ping(self.master_host, self.master_port, '/shutdown')
        self.parameter_server.join()

    def allocate_worker(self):
        worker = AsynchronousEASGDWorker(keras_model=self.master_model,
                                         features_col=self.features_column,
                                         label_col=self.label_column,
                                         rho=self.rho,
                                         communication_window=self.communication_window,
                                         learning_rate=self.learning_rate,
                                         batch_size=self.batch_size,
                                         master_host=self.master_host,
                                         master_port=self.master_port,
                                         worker_optimizer=self.worker_optimizer,
                                         loss=self.loss)

        return worker

    def service(self):
        app = Flask(__name__)

        ## BEGIN REST routes. ##################################################

        @app.route('/center_variable', methods=['GET'])
        def center_variable():
            with self.mutex:
                center_variable = self.model.get_weights()

            return pickle.dumps(center_variable, -1)

        @app.route('/update', methods=['POST'])
        def update():
            data = pickle.loads(request.data)
            variable = data['variable']
            iteration = data['iteration']
            worker_id = data['worker_id']

            with self.mutex:
                center_variable = self.model.get_weights()
                center_variable = center_variable + variable
                self.model.set_weights(center_variable)
                self.iteration += 1

            return 'OK'

        @app.route('/shutdown', methods=['GET'])
        def shutdown():
            f = request.environ.get('werkzeug.server.shutdown')
            f()

            return 'OK'

        ## END REST routes. ####################################################

        app.run(host='0.0.0.0', threaded=True, use_reloader=False)

class DOWNPOUR(AsynchronousDistributedTrainer):

    def __init__(self, keras_model, worker_optimizer, loss, num_workers=2, batch_size=32,
                 features_col="features", label_col="label", communication_window=5,
                 master_port=5000, num_epoch=1, learning_rate=0.01):
        super(DOWNPOUR, self).__init__(keras_model=keras_model, num_workers=num_workers,
                                       batch_size=batch_size, features_col=features_col,
                                       label_col=label_col, worker_optimizer=worker_optimizer,
                                       loss=loss, num_epoch=num_epoch)
        self.communication_window = communication_window
        self.master_host = determine_host_address()
        self.master_port = master_port
        self.learning_rate = learning_rate
        self.initialize_variables()

    def initialize_variables(self):
        self.iteration = 1
        self.model = deserialize_keras_model(self.master_model)

    def stop_service(self):
        rest_get_ping(self.master_host, self.master_port, '/shutdown')
        self.parameter_server.join()

    def allocate_worker(self):
        worker = DOWNPOURWorker(keras_model=self.master_model,
                                features_col=self.features_column,
                                label_col=self.label_column,
                                batch_size=self.batch_size,
                                master_host=self.master_host,
                                master_port=self.master_port,
                                learning_rate=self.learning_rate,
                                communication_window=self.communication_window,
                                worker_optimizer=self.worker_optimizer,
                                loss=self.loss)

        return worker

    def service(self):
        app = Flask(__name__)

        ## BEGIN REST routes. ##################################################

        @app.route('/center_variable', methods=['GET'])
        def center_variable():
            with self.mutex:
                center_variable = self.model.get_weights()

            return pickle.dumps(center_variable, -1)

        @app.route('/update', methods=['POST'])
        def update():
            data = pickle.loads(request.data)
            variable = data['variable']
            iteration = data['iteration']
            worker_id = data['worker_id']

            with self.mutex:
                center_variable = self.model.get_weights()
                center_variable = center_variable + variable
                self.model.set_weights(center_variable)
                self.iteration += 1

            return 'OK'

        @app.route('/shutdown', methods=['GET'])
        def shutdown():
            f = request.environ.get('werkzeug.server.shutdown')
            f()

            return 'OK'

        ## END REST routes. ####################################################

        app.run(host='0.0.0.0', threaded=True, use_reloader=False)

## END Asynchronous trainers. ##################################################

## BEGIN Synchronous trainers. #################################################

class SynchronizedDistributedTrainer(Trainer):

    def __init__(self, keras_model, worker_optimizer, loss, num_workers=2, batch_size=32,
                 features_col="features", label_col="label", num_epoch=1):
        super(SynchronizedDistributedTrainer, self).__init__(keras_model, loss, worker_optimizer)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.features_column = features_col
        self.label_column = label_col
        self.ready = False
        self.iteration = 1
        self.parameter_server = None
        self.mutex = Lock()
        self.ready_mutex = Lock()
        self.model = None

    def reset_variables(self):
        self.iteration = 1
        self.ready = False

    def set_ready(self, state):
        with self.mutex:
            self.ready = state

    def get_ready(self):
        localReady = None
        with self.ready_mutex:
            localReady = self.ready

        return localReady

    def start_service(self):
        self.parameter_server = threading.Thread(target=self.service)
        self.parameter_server.start()

    def service(self):
        raise NotImplementedError

    def stop_service(self):
        raise NotImplementedError

    def allocate_worker(self):
        raise NotImplementedError

    def train(self, data, shuffle=False):
        # Start the communication service.
        self.start_service()
        # Allocate a worker program.
        worker = self.allocate_worker()
        # Fetch the current number of partitions.
        numPartitions = data.rdd.getNumPartitions()
        # Check if we need to merge or repartition.
        if numPartitions > self.num_workers:
            data = data.coalesce(self.num_workers)
        else:
            data = data.repartition(self.num_workers)
        # Check if the data needs to be shuffled.
        if shuffle:
            data = shuffle(data)
        for i in range(0, self.num_epoch):
            self.set_ready(False)
            data.rdd.mapPartitionsWithIndex(worker.train).collect()
        # Stop the communication service.
        self.stop_service()

        return self.model

class EASGD(SynchronizedDistributedTrainer):

    def __init__(self, keras_model, worker_optimizer, loss, features_col="features", label_col="label", num_workers=2,
                 rho=5.0, learning_rate=0.01, batch_size=32, master_port=5000, num_epoch=1):
        super(EASGD, self).__init__(keras_model=keras_model, num_workers=num_workers,
                                    batch_size=batch_size, features_col=features_col,
                                    label_col=label_col, worker_optimizer=worker_optimizer,
                                    loss=loss, num_epoch=num_epoch)
        # Initialize the algorithm parameters.
        self.rho = rho
        self.learning_rate = learning_rate
        self.beta = self.num_workers * (self.learning_rate * self.rho)
        # Initialize master server parameters.
        self.master_host = determine_host_address()
        self.master_port = master_port
        # Initialize default model parameters.
        self.initialize_variables()

    def initialize_variables(self):
        # Reset the training attributes.
        self.model = deserialize_keras_model(self.master_model)
        self.variables = {}

    def stop_service(self):
        rest_get_ping(self.master_host, self.master_port, '/shutdown')
        self.parameter_server.join()

    def allocate_worker(self):
        worker = EASGDWorker(keras_model=self.master_model,
                             features_col=self.features_column,
                             label_col=self.label_column,
                             rho=self.rho,
                             learning_rate=self.learning_rate,
                             batch_size=self.batch_size,
                             master_host=self.master_host,
                             master_port=self.master_port,
                             worker_optimizer=self.worker_optimizer,
                             loss=self.loss)

        return worker

    def process_variables(self):
        center_variable = self.model.get_weights()
        temp = np.copy(center_variable)
        temp.fill(0.0)

        # Iterate through all worker variables.
        for i in range(0, self.num_workers):
            temp += (self.rho * (self.variables[i] - center_variable))
        temp /= float(self.num_workers)
        temp *= self.learning_rate
        center_variable += temp
        # Update the center variable
        self.model.set_weights(center_variable)

    def service(self):
        app = Flask(__name__)

        ## BEGIN REST routes. ##################################################

        @app.route("/center_variable", methods=['GET'])
        def center_variable():
            with self.mutex:
                center_variable = self.model.get_weights()

            return pickle.dumps(center_variable, -1)

        @app.route("/update", methods=['POST'])
        def update():
            data = pickle.loads(request.data)
            variable = data['variable']
            iteration = data['iteration']
            worker_id = data['worker_id']

            self.set_ready(False)
            # Check if the variable update is the correct iteration.
            if iteration == self.iteration:
                with self.mutex:
                    self.variables[worker_id] = variable
                    num_variables = len(self.variables)
                    # Check if the gradients of all workers are available.
                if num_variables == self.num_workers:
                    self.process_variables()
                    self.variables = {}
                    self.set_ready(True)
                    self.iteration += 1

            return 'OK'

        @app.route("/ready", methods=['POST'])
        def ready():
            data = pickle.loads(request.data)
            iteration = data['iteration']
            ready = self.get_ready()
            ready = (ready or iteration < self.iteration)

            return str(int(ready))

        @app.route("/shutdown", methods=['GET'])
        def shutdown():
            f = request.environ.get('werkzeug.server.shutdown')
            f()

            return 'OK'

        ## END REST routes. ####################################################

        app.run(host='0.0.0.0', threaded=True, use_reloader=False)

## END Trainers. ###############################################################
