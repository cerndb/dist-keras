"""Schemes module.

Module with schemes to automatize a distributed learning process. These schemes will automatically
adjust the hyperparameters to improve training performance.
"""

## BEGIN Imports. ##############################################################

import math

## END Imports. ################################################################

class Scheme(object):
    """A 'Scheme' is way to describe how a distributed optimization sequence
    should perform. For example, it is responsible for adjusting the learning
    rate of the parameter server if it notices that the loss doesn't decay.
    However, this is only one of the possible solutions. Others include the
    optimization of other hyperparameters such as the number of workers.

    # Arguments
        optimizer: trainer. A distributed optimizer.
        num_epoch: int. Total number of epoch.
        evaluation_frequency: int. Frequency of hyperparameter evaluation.
    """

    def __init__(self, optimizer, num_epoch=15, evaluation_frequency=5):
        self.optimizer = optimizer
        self.num_epoch = num_epoch
        self.evaluation_frequency = evaluation_frequency
        self.epoch_over_eval_frequency = int(self.num_epoch / self.evaluation_frequency)
        self.initialize()

    def initialize(self):
        """Initializes the hyperparameters to follow the scheme parameters."""
        self.optimizer.set_num_epoch(self.get_epoch_over_evaluation_frequency())

    def get_epoch_over_evaluation_frequency(self):
        """Returns the number of epochs per evaluation frequency."""
        return self.epoch_over_eval_frequency

    def optimize(self, training_set, validation_set):
        raise NotImplementedError


class Emperor(Scheme):
    """The 'Emporor' optimization schema will make hyperparameter changes based
    on the loss derrivatives of the validation set.

    # Arguments
        optimizer: trainer. A distributed optimizer.
        evaluate_loss: function. Function which evaluates the loss. This
                       function should accept a model, and a dataframe.
        num_epoch: int. Total number of epoch.
        evaluation_frequency: int. Frequency of hyperparameter evaluation.
    """

    def __init__(self, optimizer, evaluate_loss, num_epoch=15, evaluation_frequency=5,
                 loss_threshold=0.005):
        super(Emperor, self).__init__(optimizer, num_epoch, evaluation_frequency)
        self.previous_loss = float('inf')
        self.loss_threshold = loss_threshold
        self.evaluate_loss = evaluate_loss

    def optimize(self, training_set, validation_set):
        trained_model = None

        # Fetch the number of evaluations, to match the number of epochs.
        num_evaluations = self.get_epoch_over_evaluation_frequency() + 1
        # Iterate over the number of evaluation epochs.
        for i in range(0, num_evaluations):
            # Train the model.
            trained_model = self.optimizer.train(training_set)
            self.optimizer.set_model(trained_model)
            # Evaluate the training set, and fetch the loss.
            loss = self.evaluate_loss(trained_model, validation_set)
            print("Current loss: " + str(loss))
            dl = math.fabs(loss - self.previous_loss)
            self.previous_loss = loss
            if dl <= self.loss_threshold:
                print("Lowering learning rate.")
                print("Old learning rate: " + str(self.optimizer.get_learning_rate()))
                # Modify the learning rate.
                learning_rate = self.optimizer.get_learning_rate()
                learning_rate /= 10
                self.optimizer.set_learning_rate(learning_rate)
                print("New learning rate: "+ str(self.optimizer.get_learning_rate()))

        return trained_model
