"""Predictor classes.

Predictors take a model and will transform the Dataframe by adding a prediction column.
"""

## BEGIN Imports. ##############################################################

from distkeras.transformers import Transformer
from distkeras.utils import *

import numpy as np

## END Imports. ################################################################

class Predictor(Transformer):

    def __init__(self, keras_model):
        self.model = serialize_keras_model(keras_model)

    def predict(self, dataframe):
        raise NotImplementedError

class ModelPredictor(Predictor):
    """Takes a Keras model and adds a prediction column to the dataframe
       given a features column."""

    def __init__(self, keras_model, features_col="features", output_col="prediction"):
        super(ModelPredictor, self).__init__(keras_model)
        self.features_column = features_col
        self.output_column = output_col

    def _predict(self, iterator):
        model = deserialize_keras_model(self.model)
        for row in iterator:
            X = np.asarray([row[self.features_column]])
            Y = model.predict(X)
            v = DenseVector(Y[0])
            new_row = new_dataframe_row(row, self.output_column, v)
            yield new_row

    def predict(self, dataframe):
        return dataframe.rdd.mapPartitions(self._predict).toDF()
