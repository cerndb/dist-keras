"""Commonly used Dataframe transformers.

A transformer will "transform" a Spark dataframe from one form into
the other. For example, mapping the column to an other value, or adding
a column to a dataframe based on a collection of specified values.
"""

## BEGIN Imports. ##############################################################

import numpy as np

from distkeras.utils import new_dataframe_row
from distkeras.utils import to_one_hot_encoded_dense

from pyspark.mllib.linalg import DenseMatrix
from pyspark.mllib.linalg import DenseVector

from pyspark.sql.functions import mean
from pyspark.sql.functions import stddev_pop

## END Imports. ################################################################

class Transformer(object):
    """Interface which defines a transformer object."""

    def transform(self, dataframe):
        """Transforms the dataframe into an other dataframe.

        # Returns
            The transformed dataframe.
        """
        raise NotImplementedError


class MinMaxTransformer(Transformer):
    """Will transform every feature of an instance between a specified range.

    # Arguments
        o_min: float. Original minimum of dataset.
        o_max: float. Original maximum of dataset.
        n_min: float. New minimum of dataset.
        n_max: float. New maximum of dataset.
        input_col: string. Name of input column.
        output_col: string. Name of output column.
        is_vector. boolean. Indicates if the data element is a vector or
                            a singular value.

    # Summary
        New range: [o_min; o_max]
        Old range: [n_min; n_max]
    """

    def __init__(self, o_min, o_max, n_min, n_max, input_col, output_col, is_vector=True):
        self.o_min = float(o_min)
        self.o_max = float(o_max)
        self.n_min = float(n_min)
        self.n_max = float(n_max)
        self.scale = (self.n_max - self.n_min) / (self.o_max - self.o_min)
        self.input_column = input_col
        self.output_column = output_col
        self.is_vector = is_vector

    def _transform(self, row):
        """Rescale every instance like this:

        x' = \frac{x - min}{max - min}
        """
        if self.is_vector:
            vector = row[self.input_column].toArray()
            vector = self.scale * (vector - self.o_max) + self.n_max
            new_value = DenseVector(vector)
        else:
            value = row[self.input_column]
            new_value = self.scale * (value - self.o_max) + self.n_max
        # Construct a new row with the normalized vector.
        new_row = new_dataframe_row(row, self.output_column, new_value)

        return new_row

    def transform(self, dataframe):
        """Applies the min-max transformation to every row in the dataframe.

        # Arguments
            dataframe: dataframe. Spark Dataframe.
        """
        return dataframe.rdd.map(self._transform).toDF()


class BinaryLabelTransformer(Transformer):
    """Transformers the specified a column to a binary label, i.e., [0, 1] give
    a specific label name. Given the specified label, this transformer will generate
    [1,0], in the other case [0,1].

    # Arguments:
        input_column: string. Column name of the label identifier.
        output_column: string. Name of the new label which contains the binary label.
        label: string. Name of the label which needs to serve as 1.
    """

    def __init__(self, input_column, output_column, label):
        self.input_column = input_column
        self.output_column = output_column
        self.label = label

    def _transform(self, row):
        """Appends the desired binary label column."""
        value = row[self.input_column]
        vector = np.zeros(2)
        # Check if the name matches.
        if value == self.label:
            vector[0] = 1.0
        else:
            vector[1] = 1.0
        # Convert to a Spark DenseVector
        vector = DenseVector(vector)

        return new_dataframe_row(row, self.output_column, vector)

    def transform(self, dataframe):
        """Applies the binary label transformation to the applied dataframe.

        # Arguments
            dataframe: dataframe. Spark Dataframe.
        """
        return dataframe.rdd.map(self._transform).toDF()


class StandardTransformer(Transformer):
    """Will transform the specified columns to unit standard deviation (if specified),
    and centers the data to mean 0 (if specified).

    # Arguments
        columns: list. List of columns.
        suffix: string. Suffix name of the column after processing.
    # Note
        We assume equal probability of the rows.
    """

    def __init__(self, columns, suffix="_normalized"):
        self.columns = columns
        self.column_suffix = suffix
        self.current_column = None
        self.means = {}
        self.stddevs = {}

    def clean_mean_keys(self, means):
        """Cleans the keys of the specified dictionary (mean)."""
        new_means = {}

        for k in means:
            new_means[k[4:-1]] = means[k]

        return new_means

    def clean_stddev_keys(self, stddevs):
        """Cleans the keys of the specified dictionary (stddev)."""
        new_stddevs = {}

        for k in stddevs:
            new_stddevs[k[11:-5]] = stddevs[k]

        return new_stddevs

    def _transform(self, row):
        """Take the column, and normalize it with the computed means and std devs."""
        mean = self.means[self.current_column]
        stddev = self.stddevs[self.current_column]
        x = row[self.current_column]
        x_normalized = (x - mean) / stddev
        output_column = self.current_column + self.column_suffix
        new_row = new_dataframe_row(row, output_column, x_normalized)

        return new_row

    def transform(self, dataframe):
        """Applies standardization to the specified columns.

        # Arguments
            dataframe: dataframe. Spark Dataframe.
        """
        # Compute the means of the specified columns.
        means = [mean(x) for x in self.columns]
        means = dataframe.select(means).collect()[0].asDict()
        self.means = self.clean_mean_keys(means)
        # Compute the standard deviation of the specified columns.
        stddevs = [stddev_pop(x) for x in self.columns]
        stddevs = dataframe.select(stddevs).collect()[0].asDict()
        self.stddevs = self.clean_stddev_keys(stddevs)
        # For every feature, add a new column to the dataframe.
        for column in self.columns:
            self.current_column = column
            dataframe = dataframe.rdd.map(self._transform).toDF()

        return dataframe


class DenseTransformer(Transformer):
    """Transformes sparse vectors into dense vectors.

    # Arguments
        input_col: string. Name of the input column of the sparse vector.
        output_col: string. Name of the output column.
    """

    def __init__(self, input_col, output_col):
        self.input_column = input_col
        self.output_column = output_col

    def _transform(self, row):
        """Transforms the sparse vector to a dense vector while putting it in a new column."""
        sparse_vector = row[self.input_column]
        dense_vector = DenseVector(sparse_vector.toArray())
        new_row = new_dataframe_row(row, self.output_column, dense_vector)

        return new_row

    def transform(self, dataframe):
        """Transforms every sparse vector in the input column to a dense vector.

        # Arguments
            dataframe: dataframe. Spark Dataframe.
        # Returns
            A transformed Spark Dataframe.
        """
        return dataframe.rdd.map(self._transform).toDF()


class ReshapeTransformer(Transformer):
    """Transforms vectors into other dense shapes.

    # Note:
        Only use this transformer in the last stage of the processing pipeline.
        Since the arbitrary vector shapes will be directly passed on to the models.

    # Arguments:
        input_col: string. Name of the input column containing the vector.
        output_col: string. Name of the output column.
        shape: tuple. Shape of the matrix.
    """

    def __init__(self, input_col, output_col, shape):
        self.input_column = input_col
        self.output_column = output_col
        self.shape = shape

    def _transform(self, row):
        """Transforms the vector to a dense matrix while putting it in a new column."""
        vector = row[self.input_column]
        vector = np.asarray(vector)
        reshaped = vector.reshape(self.shape).tolist()
        new_row = new_dataframe_row(row, self.output_column, reshaped)

        return new_row

    def transform(self, dataframe):
        """Transforms every vector in the input column to a dense vector.

        # Arguments
            dataframe: dataframe. Spark Dataframe.
        # Returns
            A transformed Spark Dataframe.
        """
        return dataframe.rdd.map(self._transform).toDF()


class OneHotTransformer(Transformer):
    """Transformer which transforms an integer index into a vector using one-hot-encoding.

    # Arguments
        output_dim: int. Dimension of output vector.
        input_col: string. Name of input column.
        output_col: string. Name of output column.
    """

    def __init__(self, output_dim, input_col, output_col):
        self.input_column = input_col
        self.output_column = output_col
        self.output_dimensionality = output_dim

    def _transform(self, row):
        """Transforms every individual row.

        Only for internal use.
        """
        label = row[self.input_column]
        vector = to_one_hot_encoded_dense(label, self.output_dimensionality)
        new_row = new_dataframe_row(row, self.output_column, vector.tolist())

        return new_row

    def transform(self, dataframe):
        """Applies One-Hot encoding to every row in the dataframe.

        # Arguments
            dataframe: dataframe. A Spark Dataframe.
        # Returns
            A Spark Dataframe with one-hot encoded features.
        """
        return dataframe.rdd.map(self._transform).toDF()


class LabelIndexTransformer(Transformer):
    """Transformer which will transform a prediction vector into an integer label.

    # Arguments
        output_dim: int. Dimension of output vector.
        input_col: string. Name of the input column.
        output_col: string. Name of the output column.
        default_index: int. Default "answer".
        activation_threshold: float. Threshold of immediate activation.
    """

    def __init__(self, output_dim, input_col="prediction", output_col="prediction_index",
                 default_index=0, activation_threshold=0.55):
        self.input_column = input_col
        self.output_column = output_col
        self.output_dimensionality = output_dim
        self.activation_threshold = activation_threshold
        self.default_index = default_index

    def get_index(self, vector):
        """Returns the index with the highest value or with activation threshold."""
        max = 0.0
        max_index = self.default_index
        for index in range(0, self.output_dimensionality):
            if vector[index] >= self.activation_threshold:
                return index
            if vector[index] > max:
                max = vector[index]
                max_index = index

        return max_index

    def _transform(self, row):
        """Transforms every row by adding a "predicted index" column to the dataframe. """
        prediction = row[self.input_column]
        index = float(self.get_index(prediction))
        new_row = new_dataframe_row(row, self.output_column, index)

        return new_row

    def transform(self, dataframe):
        """Transforms the dataframe by adding a predicted index.

       # Arguments
            dataframe: dataframe. A Spark Dataframe.
        # Returns
            A Spark Dataframe with a "predicted" index.
        """
        return dataframe.rdd.map(self._transform).toDF()
