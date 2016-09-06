"""
Distributed module. This module will contain all distributed classes and
methods.
"""

## BEGIN Imports. ##############################################################

from itertools import chain
from itertools import tee

from keras.models import model_from_json
from keras.optimizers import RMSprop
from keras.utils import np_utils

from pyspark.mllib.linalg import DenseVector
from pyspark.sql import Row

import numpy as np

## END Imports. ################################################################

## BEGIN Utility functions. ####################################################

def to_vector(x, n_dim):
    vector = np.zeros(n_dim)
    vector[x] = 1.0

    return vector

## END Utility functions. ######################################################

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
                v_t = DenseVector(to_vector(label, self.output_dim).tolist())
                d = row.asDict(True)
                d[self.output_column] = v_t
                new_row = Row(**dict(d))
                rows.append(new_row)
        except TypeError:
            pass

        return iter(rows)

    def transform(self, data):
        return data.mapPartitions(self._transform)


class Predictor(Transformer):

    def __init__(self, keras_model):
        self.model = keras_model.to_json()

    def predict(self, data):
        raise NotImplementedError


class Trainer(object):

    def __init__(self, keras_model, features_col="features", label_col="label"):
        self.master_model = keras_model.to_json()
        self.features_column = features_col
        self.label_column = label_col

    def train(self, data):
        raise NotImplementedError

class SingleTrainer(Trainer):

    def __init__(self, keras_model, features_col="features", label_col="label"):
        super(SingleTrainer, self).__init__(keras_model, features_col, label_col)

    def train(self, data):
        # TODO Implement.
        pass

class EnsembleTrainer(Trainer):

    def __init__(self, keras_model, num_models=2, features_col="features",
                 label_col="label", label_transformer=None, merge_models=False):
        super(EnsembleTrainer, self).__init__(keras_model, features_col, label_col)
        self.num_models = num_models
        self.label_transformer = label_transformer
        self.merge_models = merge_models

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
        model = model_from_json(self.model)
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
