"""Evaluation module.

An evaluator will evaluate a dataframe according to specific requirements.
"""

## BEGIn Imports. ##############################################################

from distkeras.predictors import *
from distkeras.utils import *

import numpy as np

## END Imports. ################################################################

class Evaluator(object):

    def __init__(self, label_col="label", prediction_col="prediction"):
        self.label_column = label_col
        self.prediction_column = prediction_col

    def _evaluate(self, row):
        raise NotImplementedError

    def evaluate(self, dataframe):
        raise NotImplementedError

class AccuracyEvaluator(Evaluator):
    """Evaluator which will compute the prediction accuracy of a model."""

    def __init__(self, label_col="label", prediction_col="prediction"):
        # Initialize the parent structure.
        super(AccuracyEvaluator, self).__init__(features_col, label_col, prediction_col)

    def _evaluate(self, row):
        """Filters the correct predictions."""
        prediction = row[self.prediction_column]
        label = row[self.label_column]
        # Check if the prediction is equal to the label.
        if label == prediction:
            yield row

    def evaluate(self, dataframe):
        # Count the total number of instances.
        num_instances = dataframe.count()
        # Evaluate all rows.
        df = dataframe.rdd.map(self._evaluate)
        # Fetch the number of correctly guessed instances.
        validated_instances = df.count()

        return float(validated_instances) / float(num_instances)
