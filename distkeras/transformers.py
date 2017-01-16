"""Commonly used Dataframe transformers.

A transformer will "transform" a Spark dataframe from one form into
the other. For example, mapping the column to an other value, or adding
a column to a dataframe based on a collection of specified values.
"""

## BEGIN Imports. ##############################################################

from pyspark.mllib.linalg import DenseMatrix
from pyspark.mllib.linalg import DenseVector

from distkeras.utils import new_dataframe_row
from distkeras.utils import to_dense_vector

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

    # Summary
        New range: [o_min; o_max]
        Old range: [n_min; n_max]
    """

    def __init__(self, o_min, o_max, n_min, n_max, input_col, output_col):
        self.o_min = o_min
        self.o_max = o_max
        self.n_min = n_min
        self.n_max = n_max
        self.scale = (self.n_max - self.n_min) / (self.o_max - self.o_min)
        self.input_column = input_col
        self.output_column = output_col

    def _transform(self, row):
        """Rescale every instance like this:

        x' = \frac{x - min}{max - min}
        """
        vector = row[self.input_column].toArray()
        vector = self.scale * (vector - self.o_max) + self.n_max
        # Convert to a DenseVector.
        dense_vector = DenseVector(vector)
        # Construct a new row with the normalized vector.
        new_row = new_dataframe_row(row, self.output_column, dense_vector)

        return new_row

    def transform(self, dataframe):
        """Applies the min-max transformation to every row in the dataframe.

        # Arguments
            dataframe: dataframe. Spark Dataframe.
        """
        return dataframe.rdd.map(self._transform).toDF()


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
        reshaped = vector.toArray().reshape(self.shape).tolist()
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
        vector = to_dense_vector(label, self.output_dimensionality)
        new_row = new_dataframe_row(row, self.output_column, vector)

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
