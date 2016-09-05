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
## END Utility functions. ######################################################

class Trainer(object):

    def __init__(self, spark_context, keras_model):
        self.sc = spark_context
        self.master_model = keras_model.to_json()

    def train(self, data):
        raise NotImplementedError


class EnsembleTrainer(Trainer):

    def __init__(self, spark_context, keras_model, num_models=2, features_col="features", label_col="label"):
        super(EnsembleTrainer, self).__init__(spark_context, keras_model)
        self.num_models = num_models
        self.features_column = features_col
        self.label_column = label_col

    def _train(self, iterator):
        feature_iterator, label_iterator = tee(iterator, 2)
        X = np.asarray([row[self.features_column] for row in feature_iterator])
        Y = np.asarray([row[self.label_column] for row in label_iterator])
        model = model_from_json(self.master_model)
        # TODO Add compilation parameters.
        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])
        # Fit the model with the data.
        history = model.fit(X, Y, nb_epoch=1)
        partitionResult = (history, model.to_json())

        return iter([partitionResult])

    def train(self, data):
        # Repartition the data to fit the number of models.
        data = data.repartition(self.num_models)
        # Train the models.
        models = data.mapPartitions(self._train).collect()

        return models
