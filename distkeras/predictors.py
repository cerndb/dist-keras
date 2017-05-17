"""Predictors take a model and will transform the Dataframe by adding a prediction column."""

## BEGIN Imports. ##############################################################

import numpy as np

from pyspark.mllib.linalg import DenseVector

from distkeras.utils import serialize_keras_model
from distkeras.utils import deserialize_keras_model
from distkeras.utils import new_dataframe_row

## END Imports. ################################################################

class Predictor(object):
    """Abstract predictor class.

    # Arguments
        keras_model: Keras Model.
    """

    def __init__(self, keras_model):
        self.model = serialize_keras_model(keras_model)

    def predict(self, dataframe):
        """Transforms the dataframe to add a prediction.

        # Arguments
            dataframe: dataframe. Spark Dataframe.
        """
        raise NotImplementedError


class ModelPredictor(Predictor):
    """Takes a Keras model and adds a prediction column to the dataframe
       given a features column.

    # Arguments
        keras_model: Keras model.
        features_col: string. Name of the features column.
        output_col: string. Name of the prediction column.
    """

    def __init__(self, keras_model, features_col="features", output_col="prediction"):
        super(ModelPredictor, self).__init__(keras_model)
        assert isinstance(features_col, (str, list)), "'features_col' must be a string or a list of strings"
        self.features_column = [features_col] if isinstance(features_col, str) else features_col
        self.output_column = output_col

    def _predict(self, iterator):
        """Lambda method which will append a prediction column to the provided rows.

        # Arguments:
            iterator: iterator. Spark Row iterator.
        """
        model = deserialize_keras_model(self.model)
        for row in iterator:
            features = [np.asarray([row[c]]) for c in self.features_column]
            prediction = model.predict(features)
            dense_prediction = DenseVector(prediction[0])
            new_row = new_dataframe_row(row, self.output_column, dense_prediction)
            yield new_row

    def predict(self, dataframe):
        """Returns a dataframe which is the old dataframe with an additional
        prediction column.
        """
        return dataframe.rdd.mapPartitions(self._predict).toDF()
