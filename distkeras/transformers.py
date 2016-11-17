"""Commonly used Dataframe transformers.

A transformer will "transform" a Spark dataframe from one form into
the other. For example, mapping the column to an other value, or adding
a column to a dataframe based on a collection of specified values.
"""

## BEGIN Imports. ##############################################################

from distkeras.utils import *

## END Imports. ################################################################

class Transformer(object):

    def transform(self, dataframe):
        raise NotImplementedError

class OneHotTransformer(Transformer):
    """Transformer which will transform an integer label into a vector using one-hot-encoding."""

    def __init__(self, output_dim, input_col="label", output_col="label_vectorized"):
        self.input_column = input_col
        self.output_column = output_col
        self.output_dimensionality = output_dim

    def _transform(self, row):
        label = row[self.input_column]
        v = to_dense_vector(label, self.output_dimensionality)
        new_row = new_dataframe_row(row, self.output_column, v)

        return new_row

    def transform(self, dataframe):
        return dataframe.rdd.map(self._transform).toDF()

class LabelIndexTransformer(Transformer):
    """Transformer which will transform a prediction vector into an integer label."""

    def __init__(self, output_dim, input_col="prediction", output_col="prediction_index",
                 default_index=0, activation_threshold=0.55):
        self.input_column = input_col
        self.output_column = output_col
        self.output_dimensionality = output_dim
        self.activation_threshold = activation_threshold
        self.default_index = default_index

    def get_index(self, vector):
        for index in range(0, self.output_dimensionality):
            if vector[index] >= self.activation_threshold:
                return index

        return self.default_index

    def _transform(self, row):
        prediction = row[self.input_column]
        index = float(self.get_index(prediction))
        new_row = new_dataframe_row(row, self.output_column, index)

        return new_row

    def transform(self, dataframe):
        return dataframe.rdd.map(self._transform).toDF()
