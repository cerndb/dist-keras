"""
Distributed module. This module will contain all distributed classes and
methods.
"""

## BEGIN Imports. ##############################################################

from distkeras.networking import *

from flask import Flask, request

from itertools import chain
from itertools import tee

from keras.models import model_from_config
from keras.models import model_from_json
from keras.optimizers import RMSprop
from keras.utils import np_utils

from pyspark.mllib.linalg import DenseVector
from pyspark.sql import Row

from threading import Lock

import cPickle as pickle

import numpy as np

import threading

import time

import urllib2

## END Imports. ################################################################

## BEGIN Utility functions. ####################################################

def to_vector(x, n_dim):
    vector = np.zeros(n_dim)
    vector[x] = 1.0

    return vector

def new_dataframe_row(old_row, column_name, column_value):
    d = old_row.asDict(True)
    d[column_name] = column_value
    new_row = Row(**dict(d))

    return new_row

def serialize_keras_model(model):
    d = {}
    d['model'] = model.to_json()
    d['weights'] = model.get_weights()

    return d

def deserialize_keras_model(d):
    architecture = d['model']
    weights = d['weights']
    model = model_from_json(architecture)
    model.set_weights(weights)

    return model

def rest_post(host, port, endpoint, data):
    request = urllib2.Request("http://" + host + ":" + `port` + endpoint,
                              pickle.dumps(data, -1),
                              headers={'Content-Type': 'application/dist-keras'})

    return urllib2.urlopen(request).read()

def rest_get(host, port, endpoint):
    request = urllib2.Request("http://" + host + ":" + `port` + endpoint,
                              headers={'Content-Type': 'application/dist-keras'})

    return pickle.loads(urllib2.urlopen(request).read())

def rest_get_ping(host, port, endpoint):
    request = urllib2.Request("http://" + host + ":" + `port` + endpoint,
                              headers={'Content-Type': 'application/dist-keras'})
    urllib2.urlopen(request)

def uniform_weights(weights, contraints=[-0.5, 0,5]):
    raise NotImplementedError

def weights_mean(weights):
    assert(weights.shape[0] > 1)

    return np.mean(weights, axis=0)

def weights_mean_vector(weights):
    num_weights = weights.shape[0]

    # Check if the precondition has been met.
    assert(num_weights > 1)

    w = []
    for weight in weights:
        flat = np.asarray([])
        for layer in weight:
            layer = layer.flatten()
            flat = np.hstack((flat, layer))
        w.append(flat)
    w = np.asarray(w)

    return np.mean(w, axis=0)

def weights_std(weights):
    num_weights = weights.shape[0]

    # Check if the precondition has been met.
    assert(num_weights > 1)
    w = []
    for weight in weights:
        flat = np.asarray([])
        for layer in weight:
            layer = layer.flatten()
            flat = np.hstack((flat, layer))
        w.append(flat)
    w = np.asarray(w)
    std = np.std(w, axis=0)
    for i in range(0, std.shape[0]):
        if std[i] == 0.0:
            std[i] = 0.000001

    return std

## END Utility functions. ######################################################

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
                v = DenseVector(to_vector(label, self.output_dim).tolist())
                new_row = new_dataframe_row(row, self.output_column, v)
                rows.append(new_row)
        except TypeError:
            pass

        return iter(rows)

    def transform(self, data):
        return data.mapPartitions(self._transform)

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
        return data.mapPartitions(self._transform)

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
        return data.mapPartitions(self._predict)

## END Predictors. #############################################################

## BEGIN Trainers. #############################################################

class Trainer(object):

    def __init__(self, keras_model):
        self.master_model = serialize_keras_model(keras_model)

    def train(self, data):
        raise NotImplementedError

class EnsembleTrainer(Trainer):

    def __init__(self, keras_model, num_models=2, features_col="features",
                 label_col="label", label_transformer=None, merge_models=False):
        super(EnsembleTrainer, self).__init__(keras_model)
        self.num_models = num_models
        self.label_transformer = label_transformer
        self.merge_models = merge_models
        self.features_column = features_col
        self.label_column = label_col

    def merge(self, models):
        raise NotImplementedError

    def train(self, data):
        # Repartition the data to fit the number of models.
        data = data.repartition(self.num_models)
        # Allocate an ensemble worker.
        worker = EnsembleTrainerWorker(keras_model=self.master_model,
                                       features_col=self.features_column,
                                       label_col=self.label_column,
                                       label_transformer=self.label_transformer)
        # Train the models, and collect them as a list.
        models = data.mapPartitions(worker.train).collect()
        # Check if the models need to be merged.
        if self.merge_models:
            merged_model = self.merge(models)
        else:
            merged_model = None
        # Append the optional merged model to the list.
        models.append(merged_model)

        return models

class EnsembleTrainerWorker(object):

    def __init__(self, keras_model, features_col="features", label_col="label", label_transformer=None):
        self.model = keras_model
        self.features_column = features_col
        self.label_column = label_col
        self.label_transformer = label_transformer

    def train(self, iterator):
        # Deserialize the Keras model.
        model = deserialize_keras_model(self.model)
        feature_iterator, label_iterator = tee(iterator, 2)
        X = np.asarray([x[self.features_column] for x in feature_iterator])
        # Check if a label transformer is available.
        if self.label_transformer:
            Y = np.asarray([self.label_transformer(x[self.label_column]) for x in label_iterator])
        else:
            Y = np.asarray([x[self.label_column] for x in label_iterator])
        # TODO Add compilation parameters.
        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])
        # Fit the model with the data.
        history = model.fit(X, Y, nb_epoch=1)
        partitionResult = (history, model)

        return iter([partitionResult])

class SynchronizedDistributedTrainer(Trainer):

    def __init__(self, keras_model, num_workers=2, batch_size=1000,
                 features_col="features", label_col="label"):
        super(SynchronizedDistributedTrainer, self).__init__(keras_model=keras_model)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.features_column = features_col
        self.label_column = label_col
        self.ready = False
        self.iteration = 1
        self.parameter_server = None
        self.mutex = Lock()
        self.model = None

    def set_ready(self, state):
        with self.mutex:
            self.ready = state

    def get_ready(self):
        localReady = None
        with self.mutex:
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

    def train(self, data):
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
        data.rdd.mapPartitionsWithIndex(worker.train).collect()
        # Stop the communication service.
        self.stop_service()

        return self.model

class EASGD(SynchronizedDistributedTrainer):

    def __init__(self, keras_model, features_col="features", label_col="label", num_workers=2,
                 rho=5.0, learning_rate=0.01, batch_size=1000):
        super(EASGD, self).__init__(keras_model=keras_model, num_workers=num_workers,
                                    batch_size=batch_size, features_col=features_col,
                                    label_col=label_col)
        self.rho = rho
        self.learning_rate = learning_rate
        # Initialize master server parameters.
        self.master_host = determine_host_address()
        self.master_port = 5000
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
                             master_port=self.master_port)

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
                # Store the gradient of the worker.
                self.variables[worker_id] = variable
                # Check if the gradients of all workers are available.
                if len(self.variables) == self.num_workers:
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

class DPGO(SynchronizedDistributedTrainer):

    def __init__(self, keras_model, num_workers=2, batch_size=1000,
                 features_col="features", label_col="label"):
        super(DPGO, self).__init__(keras_model=keras_model, num_workers=num_workers,
                                   batch_size=batch_size, features_col=features_col,
                                   label_col=label_col)
        self.master_host = determine_host_address()
        self.master_port = 5000
        self.initialize_variables()

    def initialize_variables(self):
        self.model = deserialize_keras_model(self.master_model)
        self.variables = {}
        self.mean = None
        self.mean_vector = None
        self.cov = None

    def stop_service(self):
        rest_get_ping(self.master_host, self.master_port, '/shutdown')
        self.parameter_server.join()

    def allocate_worker(self):
        worker = DPGOWorker(keras_model=self.master_model,
                            batch_size=self.batch_size,
                            features_col=self.features_column,
                            label_col=self.label_column,
                            master_host=self.master_host,
                            master_port=self.master_port)

        return worker

    def process_variables(self):
        with self.mutex:
            # Compute the mean of the variables.
            variables = np.asarray(self.variables.values())
            self.mean = weights_mean(variables)
            self.mean_vector = weights_mean_vector(variables)
            self.std_vector = weights_std(variables)
            self.std_vector /= 3.0

    def service(self):
        app = Flask(__name__)

        ## BEGIN REST routes. ##################################################

        @app.route("/distribution", methods=['GET'])
        def distribution():
            data = {}
            data['mean'] = self.mean
            data['mean_vector'] = self.mean_vector
            data['std_vector'] = self.std_vector

            return pickle.dumps(data, -1)

        @app.route("/update", methods=['POST'])
        def update():
            data = pickle.loads(request.data)
            variable = data['variable']
            iteration = data['iteration']
            worker_id = data['worker_id']
            self.set_ready(False)
            if iteration == self.iteration:
                self.variables[worker_id] = variable
                if len(self.variables) == self.num_workers:
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
            # Set the output model to the mean.
            self.model.set_weights(self.mean)

            return 'OK'

        ## END REST routers. ###################################################

        app.run(host='0.0.0.0', threaded=True, use_reloader=False)

## END Trainers. ###############################################################

## BEGIN Workers. ##############################################################


class DPGOWorker(object):

    def __init__(self, keras_model, features_col="features", label_col="label", batch_size=1000,
                 master_host="localhost", master_port=5000):
        self.model = keras_model
        self.features_column = features_col
        self.label_column = label_col
        self.master_host = master_host
        self.master_port = master_port
        self.batch_size = batch_size
        self.iteration = 1
        self.mean = None
        self.mean_vector = None
        self.std_vector = None

    def master_is_ready(self):
        data = {}
        data['iteration'] = self.iteration
        master_ready = int(rest_post(self.master_host, self.master_port, "/ready", data))

        return master_ready == 1

    def master_fetch_distribution(self):
        data = rest_get(self.master_host, self.master_port, "/distribution")
        mean = data['mean']
        mean_vector = data['mean_vector']
        std_vector = data['std_vector']
        self.mean = mean
        self.mean_vector = mean_vector
        self.std_vector = std_vector

    def master_send_variable(self, worker_id, variable):
        data = {}
        data['worker_id'] = worker_id
        data['iteration'] = self.iteration
        data['variable'] = variable
        rest_post(self.master_host, self.master_port, "/update", data)

    def convert_to_weights(self, model, vector):
        weights_structure = model.get_weights()
        transformed_weights = []
        data_index = 0
        for layer in weights_structure:
            # Check if the current layer is a trainable layer.
            if len(layer.shape) > 1:
                layer = np.zeros(layer.shape)
                for i in range(0, layer.shape[0]):
                    for j in range(0, layer.shape[1]):
                        layer[i][j] = vector[data_index]
                        data_index += 1
            transformed_weights.append(layer)

        return np.asarray(transformed_weights)

    def sample_distribution(self):
        sample = np.zeros(len(self.mean_vector))
        for i in range(0, len(sample)):
            sample[i] = np.random.normal(self.mean_vector[i], self.std_vector[i])

        return sample

    def train(self, index, iterator):
        # Deserialize the Keras model.
        model = deserialize_keras_model(self.model)
        # TODO Random weight initialization.
        # Compile the model.
        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])
        try:
            while True:
                batch = [next(iterator) for _ in range(self.batch_size)]
                feature_iterator, label_iterator = tee(batch, 2)
                X = np.asarray([x[self.features_column] for x in feature_iterator])
                Y = np.asarray([x[self.label_column] for x in label_iterator])
                model.fit(X, Y, nb_epoch=1)
                W = np.asarray(model.get_weights())
                self.master_send_variable(index, W)
                while not self.master_is_ready():
                    time.sleep(0.2)
                self.iteration += 1
                self.master_fetch_distribution()
                sample = self.sample_distribution()
                # Sample an instance from the distribution and convert.
                weights_sample = self.convert_to_weights(model, sample)
                model.set_weights(weights_sample)
        except StopIteration:
            pass

        return iter([])

class EASGDWorker(object):

    def __init__(self, keras_model, features_col="features", label_col="label", batch_size=1000,
                 rho=5, learning_rate=0.01, master_host="localhost", master_port=5000):
        self.model = keras_model
        self.features_column = features_col
        self.label_column = label_col
        self.master_host = master_host
        self.master_port = master_port
        self.master_variable = None
        self.batch_size = batch_size
        self.rho = rho
        self.iteration = 1
        self.learning_rate = learning_rate

    def master_send_variable(self, worker_id, variable):
        data = {}
        data['worker_id'] = worker_id
        data['iteration'] = self.iteration
        data['variable'] = variable
        rest_post(self.master_host, self.master_port, "/update", data)

    def master_is_ready(self):
        data = {}
        data['iteration'] = self.iteration
        master_ready = int(rest_post(self.master_host, self.master_port, "/ready", data))

        return master_ready == 1

    def fetch_center_variable(self):
        self.center_variable = np.asarray(rest_get(self.master_host, self.master_port, "/center_variable"))

    def train(self, index, iterator):
        # Deserialize the Keras model.
        model = deserialize_keras_model(self.model)
        # Initialize the model weights with a constrainted uniform distribution.
        #weights = uniform_weights(model.get_weights(), [-5, 5])
        #model.set_weights(weights)
        # Compile the model.
        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])
        try:
            while True:
                self.fetch_center_variable()
                batch = [next(iterator) for _ in range(self.batch_size)]
                feature_iterator, label_iterator = tee(batch, 2)
                X = np.asarray([x[self.features_column] for x in feature_iterator])
                Y = np.asarray([x[self.label_column] for x in label_iterator])
                W1 = np.asarray(model.get_weights())
                model.fit(X, Y, nb_epoch=1)
                W2 = np.asarray(model.get_weights())
                gradient = W2 - W1
                self.master_send_variable(index, W2)
                W = W1 - self.learning_rate * (gradient + self.rho * (W1 - self.center_variable))
                model.set_weights(W)
                while not self.master_is_ready():
                    time.sleep(0.2)
                self.iteration += 1
        except StopIteration:
            pass

        return iter([])

## END Workers. ################################################################
