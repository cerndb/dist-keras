"""Model optimizers. Depending on the implementation, these classes will optimize the
Keras model in a distributed manner (with exception of the SingleTrainer)."""

## BEGIN Imports. ##############################################################

import numpy as np

import threading

import time

from distkeras.parameter_servers import ADAGParameterServer
from distkeras.parameter_servers import DeltaParameterServer
from distkeras.parameter_servers import DynSGDParameterServer
from distkeras.parameter_servers import ExperimentalParameterServer

from distkeras.utils import deserialize_keras_model
from distkeras.utils import history_executor
from distkeras.utils import history_executors_average
from distkeras.utils import pickle_object
from distkeras.utils import serialize_keras_model
from distkeras.utils import set_keras_base_directory
from distkeras.utils import unpickle_object

from distkeras.networking import determine_host_address

from distkeras.workers import ADAGWorker
from distkeras.workers import AEASGDWorker
from distkeras.workers import DOWNPOURWorker
from distkeras.workers import DynSGDWorker
from distkeras.workers import ExperimentalWorker
from distkeras.workers import EAMSGDWorker
from distkeras.workers import SequentialWorker

from keras import backend as K

## END Imports. ################################################################

class Trainer(object):
    """Abstract trainer class. This class provides all base functionality which
    all optimizers need to implement.

    # Arguments
        keras_model: Keras model.
        loss: string. String representing the loss.
              See: https://keras.io/objectives/
        worker_optimizer: string. String representing worker optimizer.
                          See https://keras.io/optimizers/
        metrics: list of strings representing model evaluation metrics. Default is ["accuracy"].
                 See: https://keras.io/metrics/
    """

    def __init__(self, keras_model, loss, worker_optimizer, metrics=["accuracy"]):
        set_keras_base_directory()
        self.master_model = serialize_keras_model(keras_model)
        self.loss = loss
        self.worker_optimizer = worker_optimizer
        self.metrics = metrics
        self.history = []
        self.training_time_start = 0
        self.training_time_end = 0
        self.training_time = 0
        self.max_mini_batches_prefetch = 100

    def set_max_prefetch(self, max_mini_batches):
        """Sets the maximum amount of mini-batches that can be prefetched by a worker."""
        self.max_mini_batches_prefetch = max_mini_batches

    def set_model(self, model):
        """Sets the master model to be used by the trainer."""
        self.master_model = serialize_keras_model(model)

    def record_training_start(self):
        """Records the start of the training.

        This private function is called when the training process starts.
        """
        self.training_time = 0
        self.training_time_start = time.time()

    def record_training_end(self):
        """Records the end of the traing.

        This private function is called when the training process is terminated.
        """
        self.training_time_end = time.time()
        self.training_time = self.training_time_end - self.training_time_start

    def get_training_time(self):
        """Returns the told training time."""
        return self.training_time

    def get_history(self):
        """Returns all history object aggregated during training."""
        return self.history

    def get_averaged_history(self):
        """Returns the averaged history of the center variable."""
        return history_executors_average(self.history)

    def get_executor_history(self, executor_id):
        """Returns the history of a specific executor."""
        return history_executor(self.history, executor_id)

    def train(self, dataframe, shuffle=False):
        """Trains the specified model using the specified dataframe.

        # Arguments
            dataframe: dataframe. Spark Dataframe
            shuffle: boolean. Tells to shuffle the dataframe before training.
                     Warning: this will tell Spark to shuffle all partitions over
                     the network. It is recommended to shuffle the dataset before
                     training and store it.
        """
        raise NotImplementedError

    def serialize(self):
        return pickle_object(self)


class SingleTrainer(Trainer):
    """An optimizer which will train a network on a single machine.

    # Arguments
        keras_model: model. Keras model to train.
        worker_optimizer: string. String representing worker optimizer.
                          See https://keras.io/optimizers/
        loss: string. String representing the loss.
              See: https://keras.io/objectives/
        metrics: list of strings representing model evaluation metrics. Default is ["accuracy"].
                 See: https://keras.io/metrics/
        features_col: string or list of strings. Name(s) of the features column(s).
        label_col: string. Name of the label column.
        num_epoch: int. Number of epochs.
        batch_size: int. Mini-batch size.
    """

    def __init__(self, keras_model, worker_optimizer, loss, metrics=["accuracy"], features_col="features",
                 label_col="label", num_epoch=1, batch_size=32):
        super(SingleTrainer, self).__init__(keras_model, loss, worker_optimizer, metrics)
        self.features_column = features_col
        self.label_column = label_col
        self.num_epoch = num_epoch
        self.batch_size = batch_size

    def allocate_worker(self):
        """Allocates a worker for the Single Trainer instance.

        Only for internal use.
        """
        worker = SequentialWorker(model=self.master_model, features_col=self.features_column,
                                  label_col=self.label_column, batch_size=self.batch_size,
                                  optimizer=self.worker_optimizer, loss=self.loss, metrics = self.metrics)

        return worker

    def train(self, dataframe, shuffle=False):
        """See distkeras.trainers.Trainer.train

        # Arguments
            dataframe: dataframe. Spark Dataframe
            shuffle: boolean. Tells to shuffle the dataframe before training.
                     Warning: this will tell Spark to shuffle all partitions over
                     the network. It is recommended to shuffle the dataset before
                     training and store it.
        """
        # Assign the dataset.
        dataset = dataframe
        # Build the dataset with the number of epochs.
        for i in range(0, self.num_epoch):
            dataset = dataset.unionAll(dataframe)
        # Check if the data needs to be shuffled.
        if shuffle:
            dataset = shuffle(dataset)
        # Collect the dataset on a single worker node.
        dataset = dataset.coalesce(1)
        # Cache the dataset.
        dataset.cache()
        # Allocate a worker.
        worker = self.allocate_worker()
        # Set the maximum number of mini-batches.
        worker.set_max_prefetch(self.max_mini_batches_prefetch)
        # Start recording training time.
        self.record_training_start()
        # Fetch the trained model.
        self.master_model = dataset.rdd.mapPartitionsWithIndex(worker.train).collect()[0]
        # Stop recording of training time.
        self.record_training_end()

        return deserialize_keras_model(self.master_model)


class AveragingTrainer(Trainer):
    """A trainer which implements a data parallel technique using model averaging.

    In this implementation, the model replicas are averages after every epoch.
    # Arguments
        keras_model: model. Keras model to train.
        worker_optimizer: string. String representing worker optimizer.
                          See https://keras.io/optimizers/
        loss: string. String representing the loss.
              See: https://keras.io/objectives/
        metrics: list of strings representing model evaluation metrics. Default is ["accuracy"].
                 See: https://keras.io/metrics/
        features_col: string or list of strings. Name(s) of the features column(s).
        label_col: string. Name of the label column.
        num_epoch: int. Number of epochs.
        batch_size: int. Mini-batch size.
        num_workers: int. Number of model replicas to train in parallel.
    """

    def __init__(self, keras_model, worker_optimizer, loss, metrics=["accuracy"], features_col="features",
                 label_col="label", num_epoch=1, batch_size=32, num_workers=2):
        super(AveragingTrainer, self).__init__(keras_model, loss, worker_optimizer, metrics)
        self.features_column = features_col
        self.label_column = label_col
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.parameter_buffer = np.asarray(keras_model.get_weights())
        self.parameter_buffer.fill(0.0)

    def average_models(self, models):
        """Averages the specified list of Keras models, and assigns the
        averaged model as the master model.

        # Arguments:
            models: list. A list of serialized Keras models.
        """
        num_models = len(models)
        # Get all weights of the models.
        for i in range(0, num_models):
            weights = np.asarray(deserialize_keras_model(models[i]).get_weights())
            self.parameter_buffer += weights
        # Average the parameters.
        self.parameter_buffer /= num_models
        temp_model = deserialize_keras_model(self.master_model)
        temp_model.set_weights(self.parameter_buffer)
        self.master_model = serialize_keras_model(temp_model)


    def allocate_worker(self):
        """Allocates the AveragingWorker for internal use."""
        worker = SequentialWorker(model=self.master_model, features_col=self.features_column,
                                  label_col=self.label_column, batch_size=self.batch_size,
                                  optimizer=self.worker_optimizer, loss=self.loss, metrics = self.metrics)

        return worker

    def train(self, dataframe, shuffle=False):
        """Applies model averaging to the model replicas distributed over the specified
        number of Spark executors.

        # Arguments
            dataframe: dataframe: A Spark Dataframe containing the dataset.
            shuffle: boolean. Tells to shuffle the dataframe before training.
                     Warning: this will tell Spark to shuffle all partitions over
                     the network. It is recommended to shuffle the dataset before
                     training and store it.
        """
        # Repartition the data in order to fit the number of workers.
        num_partitions = dataframe.rdd.getNumPartitions()
        # Check if the dataset needs to be shuffled.
        if shuffle:
            dataframe = shuffle(dataframe)
        # Check if we need to repartition the dataframe.
        if num_partitions > self.num_workers:
            dataframe = dataframe.coalesce(self.num_workers)
        else:
            dataframe = dataframe.repartition(self.num_workers)
        # Start the training procedure.
        self.record_training_start()
        for i in range(0, self.num_epoch):
            worker = self.allocate_worker()
            # Set the maximum number of mini-batches.
            worker.set_max_prefetch(self.max_mini_batches_prefetch)
            models = dataframe.rdd.mapPartitionsWithIndex(worker.train).collect()
            self.average_models(models)
        # End the training procedure.
        self.record_training_end()

        return deserialize_keras_model(self.master_model)


class EnsembleTrainer(Trainer):
    """Utility trainer which will train ensemble methods in parallel.

    # Arguments
        keras_model: model. Keras model to train.
        worker_optimizer: string. String representing worker optimizer.
                          See https://keras.io/optimizers/
        loss: string. String representing the loss.
              See: https://keras.io/objectives/
        metrics: list of strings representing model evaluation metrics. Default is ["accuracy"].
                 See: https://keras.io/metrics/
        features_col: string or list of strings. Name(s) of the features column(s).
        label_col: string. Name of the label column.
        batch_size: int. Mini-batch size.
        num_ensembles: int. Number of ensembles to train.
    # Note
        This will note employ a data-parallell approach for the ensembles.
    """

    def __init__(self, keras_model, worker_optimizer, loss, metrics=["accuracy"], features_col="features",
                 label_col="label", batch_size=32, num_ensembles=2):
        super(EnsembleTrainer, self).__init__(keras_model, loss, worker_optimizer, metrics)
        self.features_column = features_col
        self.label_column = label_col
        self.batch_size = batch_size
        self.num_ensembles = num_ensembles

    def allocate_worker(self):
        """Allocates the EnsembleWorker for internal use."""
        worker = SequentialWorker(model=self.master_model, features_col=self.features_column,
                                  label_col=self.label_column, batch_size=self.batch_size,
                                  optimizer=self.worker_optimizer, loss=self.loss, metrics=self.metrics)

        return worker

    def train(self, dataframe, shuffle=False):
        """Trains the specified number of ensemble models using the specified dataframe.

        # Arguments
            dataframe: dataframe. A Spark Dataframe containing the dataset.
            shuffle: boolean. Tells to shuffle the dataframe before training.
                     Warning: this will tell Spark to shuffle all partitions over
                     the network. It is recommended to shuffle the dataset before
                     training and store it.
        """
        # Allocate a worker.
        worker = self.allocate_worker()
        # Set the maximum number of mini-batches.
        worker.set_max_prefetch(self.max_mini_batches_prefetch)
        # Repartition in order to fit the number of workers.
        num_partitions = dataframe.rdd.getNumPartitions()
        # Check if the dataframe needs to be shuffled before training.
        if shuffle:
            dataframe = shuffle(dataframe)
        # Check if we need to repartition the dataframe.
        if num_partitions > self.num_workers:
            dataframe = dataframe.coalesce(self.num_workers)
        else:
            dataframe = dataframe.repartition(self.num_workers)
        # Start the training procedure.
        self.record_training_start()
        # Train the models in parallel.
        models = dataframe.rdd.mapPartitionsWithIndex(worker.train).collect()
        # End the training procedure.
        self.record_training_end()

        return models


class DistributedTrainer(Trainer):
    """Abstract class which describes the properties of a distributed optimizer.

    # Arguments
        keras_model: model. Keras model to train.
        worker_optimizer: string. String representing worker optimizer.
                          See https://keras.io/optimizers/
        loss: string. String representing the loss.
              See: https://keras.io/objectives/
        metrics: list of strings representing model evaluation metrics. Default is ["accuracy"].
                 See: https://keras.io/metrics/
        features_col: string or list of strings. Name(s) of the features column(s).
        label_col: string. Name of the label column.
        num_epoch: int. Number of epochs.
        batch_size: int. Mini-batch size.
        num_workers: int. Number of distributed workers.
    """

    def __init__(self, keras_model, worker_optimizer, loss, metrics=["accuracy"], num_workers=2, batch_size=32,
                 features_col="features", label_col="label", num_epoch=1):
        super(DistributedTrainer, self).__init__(keras_model, loss, worker_optimizer, metrics)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.features_column = features_col
        self.label_column = label_col
        self.num_epoch = num_epoch
        self.parameter_server = None
        self.parameter_server_thread = None
        self.master_host = determine_host_address()
        self.master_port = 5000
        self.learning_rate = 1.0

    def set_minibatch_size(self, size):
        """Sets the size of the mini-batch."""
        self.batch_size = size

    def get_minibatch_size(self):
        """Returns the size of the mini-batch."""
        return self.batch_size

    def get_features_column(self):
        """Returns the name of the features column."""
        return self.features_column

    def get_label_column(self):
        """Returns the name of the label column."""
        return self.label_column

    def get_learning_rate(self):
        """Returns the learning rate of the worker which can be tuned by
        the parameter server, or optimization scheme.

        Note: this learning rate is independent of the learning rate of the optimizer.
        """
        return self.learning_rate

    def set_learning_rate(self, learning_rate):
        """Sets the learning rate which can be tuned by the parameter server,
        or optimization scheme.

        Note: this learning rate is independent of the learning rate of the optimizer.
        """
        self.learning_rate = learning_rate

    def set_num_epoch(self, num_epoch):
        """Sets the number of epochs."""
        self.num_epoch = num_epoch

    def get_num_epoch(self):
        """Returns the number of epochs."""
        return self.num_epoch

    def allocate_worker(self):
        """Allocates the worker implementation.

        Implement this method in subclasses.
        """
        raise NotImplementedError

    def set_master(self, master):
        """Sets the master address of the parameter server."""
        self.master_host = master

    def determine_new_master(self):
        """Sets the new master address to the current host."""
        self.master_host = determine_host_address()

    def allocate_parameter_server(self):
        """Allocates the parameter server.

        If an other type of parameter server is required, you can overwrite
        this implementation.
        """
        parameter_server = DeltaParameterServer(self.master_model, self.master_port)

        return parameter_server

    def set_num_workers(self, num_workers):
        """Sets the number of parallel workers to use."""
        self.num_workers = num_workers

    def get_num_workers(self):
        """Returns the number of parallel workers."""
        return self.num_workers

    def num_updates(self):
        """Returns the number of model updates the parameter server performed."""
        return self.parameter_server.num_updates()

    def service(self):
        """Executes the parameter server service."""
        self.parameter_server.start()
        self.parameter_server.initialize()
        self.parameter_server.run()

    def stop_service(self):
        """Stops the parameter server service."""
        self.parameter_server.stop()
        self.parameter_server_thread.join()
        self.parameter_server_thread = None

    def start_service(self):
        """Starts the parameter server service."""
        # Check if a parameter server thread is already allocated.
        if not self.parameter_server_thread is None:
            # Stop the parameter server service.
            self.stop_service()
        # Allocate a new parameter service thread.
        self.parameter_server_thread = threading.Thread(target=self.service)
        self.parameter_server_thread.start()

    def train(self, dataframe, shuffle=False):
        """Training procedure of a distributed optimization process.

        # Arguments
            dataframe: dataframe. Spark Dataframe
            shuffle: boolean. Tells to shuffle the dataframe before training.
                     Warning: this will tell Spark to shuffle all partitions over
                     the network. It is recommended to shuffle the dataset before
                     training and store it.
        """
        # Check if a parameter server has been allocated.
        if self.parameter_server is not None:
            # Cleanup the old parameter server.
            self.parameter_server.stop()
            self.parameter_server = None
        # Allocate the parameter server.
        self.parameter_server = self.allocate_parameter_server()
        # Start the communication service.
        self.start_service()
        # Allocate a worker.
        worker = self.allocate_worker()
        # Set the maximum number of mini-batches.
        worker.set_max_prefetch(self.max_mini_batches_prefetch)
        # Repartition in order to fit the number of workers.
        num_partitions = dataframe.rdd.getNumPartitions()
        # Assign the dataset.
        dataset = dataframe
        # Build a dataset which fits the number of epochs.
        for i in range(1, self.num_epoch):
            dataset = dataset.unionAll(dataframe)
        # Check if the dataframe needs to be shuffled before training.
        if shuffle:
            dataset = shuffle(dataset)
        # Check if we need to repartition the dataframe.
        if num_partitions > self.num_workers:
            dataset = dataset.coalesce(self.num_workers)
        else:
            dataset = dataset.repartition(self.num_workers)
        # Cache the dataset.
        dataset.cache()
        # Start the training procedure.
        self.record_training_start()
        # Iterate through the epochs.
        self.history = dataset.rdd.mapPartitionsWithIndex(worker.train).collect()
        # End the training procedure.
        self.record_training_end()
        # Stop the communication service.
        self.stop_service()

        return self.parameter_server.get_model()


class AsynchronousDistributedTrainer(DistributedTrainer):
    """Abstract class for an asynchronous distributed trainer.

    This trainer also allows us to set a parallelism factor. This parallelism factor allows
    us to further parallelize the Spark job. For example, imagine having n machines optimizing
    a model in an asynchronous distributed setting. If for some, but likely reason, some machines
    are performing worse compared to others. It will cause the complete learning procedure to be
    stuck on this one particular machine since every machine will be assigned a single partition.
    In order to resolve this, we added a parallelization factor. This factor indicates the ratio
    of the number of jobs per machine (executor). For small datasets, we recommend that this factor
    is set to 1. However, this effect really is prominent when the dataset is large. In this case
    we recommend that the ratio is 2 or 3.

    # Arguments
        keras_model: model. Keras model to train.
        worker_optimizer: string. String representing worker optimizer.
                          See https://keras.io/optimizers/
        loss: string. String representing the loss.
              See: https://keras.io/objectives/
        metrics: list of strings representing model evaluation metrics. Default is ["accuracy"].
                 See: https://keras.io/metrics/
        features_col: string or list of strings. Name(s) of the features column(s).
        label_col: string. Name of the label column.
        num_epoch: int. Number of epochs.
        batch_size: int. Mini-batch size.
        num_workers: int. Number of distributed workers.

    # Note
        By default, the parallelization factor is set to 1.
    """

    def __init__(self, keras_model, worker_optimizer, loss, metrics=["accuracy"], num_workers=2, batch_size=32,
                 features_col="features", label_col="label", num_epoch=1):
        super(AsynchronousDistributedTrainer, self).__init__(keras_model, worker_optimizer, loss, metrics, 
                                                             num_workers, batch_size, features_col,
                                                             label_col, num_epoch)
        # Initialize asynchronous methods variables.
        self.parallelism_factor = 1

    def allocate_worker(self):
        """Allocates the worker implementation.

        Implement this method in subclasses.
        """
        raise NotImplementedError

    def set_parallelism_factor(self, factor):
        """Sets the parallelization factor.

        # Arguments
            factor: int. The new parallelization factor.
        """
        self.parallelism_factor = factor

    def get_parallelism_factor(self):
        """Returns the parallelization factor."""
        return self.parallelism_factor

    def train(self, dataframe, shuffle=False):
        """Training procedure of an asynchronous distributed optimization process.

        # Arguments
            dataframe: dataframe. Spark Dataframe
            shuffle: boolean. Tells to shuffle the dataframe before training.
                     Warning: this will tell Spark to shuffle all partitions over
                     the network. It is recommended to shuffle the dataset before
                     training and store it.
        """
        # Check if a parameter server has been allocated.
        if self.parameter_server is not None:
            # Cleanup the old parameter server.
            self.parameter_server.stop()
            self.parameter_server = None
        # Allocate the parameter server.
        self.parameter_server = self.allocate_parameter_server()
        # Start the communication service.
        self.start_service()
        # Allocate a worker.
        worker = self.allocate_worker()
        # Set the maximum number of mini-batches.
        worker.set_max_prefetch(self.max_mini_batches_prefetch)
        # Repartition in order to fit the number of workers.
        num_partitions = dataframe.rdd.getNumPartitions()
        # Assign the dataset.
        dataset = dataframe
        # Build the dataset with the number of epochs.
        for i in range(1, self.num_epoch):
            dataset = dataset.unionAll(dataframe)
        # Check if the dataframe needs to be shuffled before training.
        if shuffle:
            dataset = shuffle(dataset)
        # Indicate the parallelism (number of worker times parallelism factor).
        parallelism = self.parallelism_factor * self.num_workers
        # Check if we need to repartition the dataframe.
        if num_partitions > parallelism:
            dataset = dataset.coalesce(parallelism)
        else:
            dataset = dataset.repartition(parallelism)
        # Start the training procedure.
        self.record_training_start()
        # Iterate through the epochs.
        self.history = dataset.rdd.mapPartitionsWithIndex(worker.train).collect()
        # End the training procedure.
        self.record_training_end()
        # Stop the communication service.
        self.stop_service()

        return self.parameter_server.get_model()


class AEASGD(AsynchronousDistributedTrainer):
    """Asynchronous Elastic Averaging SGD optimizer.
    Introduced by Zhang et al.
    https://arxiv.org/pdf/1412.6651.pdf
    # Arguments
        keras_model: model. Keras model to train.
        worker_optimizer: string. String representing worker optimizer.
                          See https://keras.io/optimizers/
        loss: string. String representing the loss.
              See: https://keras.io/objectives/
        metrics: list of strings representing model evaluation metrics. Default is ["accuracy"].
                 See: https://keras.io/metrics/
        features_col: string or list of strings. Name(s) of the features column(s).
        label_col: string. Name of the label column.
        num_epoch: int. Number of epochs.
        batch_size: int. Mini-batch size.
        num_workers: int. Number of distributed workers.
        communication_window: int. Staleness parameter.
                              This parameter describes the number of mini-batches that will be
                              computed before updating the center variable. For EASGD based
                              algorithms we recommend large communication windows.
        learning_rate: float. Learning rate.
        rho: float. Elastic "exploration" variable.
                    Higher values mean that the model is allowed to "explore" its surroundings.
                    Smaller values are correlated with less exploration. We use the value
                    recommend by the authors.
    """

    def __init__(self, keras_model, worker_optimizer, loss, metrics=["accuracy"], num_workers=2, batch_size=32,
                 features_col="features", label_col="label", num_epoch=1, communication_window=32,
                 rho=5.0, learning_rate=0.1):
        super(AEASGD, self).__init__(keras_model, worker_optimizer, loss, metrics, num_workers,
                                     batch_size, features_col, label_col, num_epoch)
        self.communication_window = communication_window
        self.rho = rho
        self.learning_rate = learning_rate

    def allocate_worker(self):
        """Allocates the asynchronous EASGD worker."""
        # Allocate a AEASGD worker.
        worker = AEASGDWorker(self.master_model, self.worker_optimizer, self.loss, self.metrics,
                              self.features_column, self.label_column, self.batch_size,
                              self.master_host, self.master_port, self.rho, self.learning_rate,
                              self.communication_window)

        return worker


class DOWNPOUR(AsynchronousDistributedTrainer):
    """DOWNPOUR Optimizer.

    Asynchronous data-parallel optimizer introduced by Dean et al.
    http://static.googleusercontent.com/media/research.google.com/en/archive/large_deep_networks_nips2012.pdf

    # Arguments
        keras_model: model. Keras model to train.
        worker_optimizer: string. String representing worker optimizer.
                          See https://keras.io/optimizers/
        loss: string. String representing the loss.
              See: https://keras.io/objectives/
        metrics: list of strings representing model evaluation metrics. Default is ["accuracy"].
                 See: https://keras.io/metrics/
        features_col: string or list of strings. Name(s) of the features column(s).
        label_col: string. Name of the label column.
        num_epoch: int. Number of epochs.
        batch_size: int. Mini-batch size.
        num_workers: int. Number of distributed workers.
        communication_window: int. Staleness parameter.
                              This parameter describes the number of mini-batches that will be
                              computed before updating the center variable. For DOWNPOUR we
                              recommend small communication windows.
        learning_rate: float. Learning rate.
    """

    def __init__(self, keras_model, worker_optimizer, loss, metrics=["accuracy"], num_workers=2, batch_size=32,
                 features_col="features", label_col="label", num_epoch=1, communication_window=5):
        super(DOWNPOUR, self).__init__(keras_model, worker_optimizer, loss, metrics, num_workers,
                                       batch_size, features_col, label_col, num_epoch)
        self.communication_window = communication_window

    def allocate_worker(self):
        """Allocates the DOWNPOUR worker."""
        # Allocate DOWNPOUR worker.
        worker = DOWNPOURWorker(self.master_model, self.worker_optimizer, self.loss, self.metrics,
                                self.features_column, self.label_column, self.batch_size,
                                self.master_host, self.master_port, self.communication_window)

        return worker


class EAMSGD(AsynchronousDistributedTrainer):
    """Asynchronous Elastic Averaging w/ Momentum SGD optimizer.

    Introduced by Zhang et al.
    https://arxiv.org/pdf/1412.6651.pdf

    # Arguments
        keras_model: model. Keras model to train.
        worker_optimizer: string. String representing worker optimizer.
                          See https://keras.io/optimizers/
        loss: string. String representing the loss.
              See: https://keras.io/objectives/
        metrics: list of strings representing model evaluation metrics. Default is ["accuracy"].
                 See: https://keras.io/metrics/
        features_col: string or list of strings. Name(s) of the features column(s).
        label_col: string. Name of the label column.
        num_epoch: int. Number of epochs.
        batch_size: int. Mini-batch size.
        num_workers: int. Number of distributed workers.
        communication_window: int. Staleness parameter.
                              This parameter describes the number of mini-batches that will be
                              computed before updating the center variable. For EASGD based
                              algorithms we recommend large communication windows.
        learning_rate: float. Learning rate.
        rho: float. Elastic "exploration" variable.
                    Higher values mean that the model is allowed to "explore" its surroundings.
                    Smaller values are correlated with less exploration. We use the value
                    recommend by the authors.
        momentum: float. Momentum term.
    """

    def __init__(self, keras_model, worker_optimizer, loss, metrics=["accuracy"], num_workers=2, batch_size=32,
                 features_col="features", label_col="label", num_epoch=1, communication_window=32,
                 rho=5.0, learning_rate=0.1, momentum=0.9):
        super(EAMSGD, self).__init__(keras_model, worker_optimizer, loss, metrics, num_workers,
                                     batch_size, features_col, label_col, num_epoch)
        self.communication_window = communication_window
        self.rho = rho
        self.learning_rate = learning_rate
        self.momentum = momentum

    def allocate_worker(self):
        """Allocates the asynchronous EAMSGD worker."""
        # Allocate a EAMSGD REST worker.
        worker = EAMSGDWorker(self.master_model, self.worker_optimizer, self.loss, self.metrics,
                              self.features_column, self.label_column, self.batch_size,
                              self.master_host, self.master_port, self.rho, self.learning_rate,
                              self.momentum, self.communication_window)

        return worker


class ADAG(AsynchronousDistributedTrainer):
    """Asynchronous Distributed Adaptive Gradient (Stochastic Gradient Descent).

    Introduced by Hermans et al.

    # Arguments:
        keras_model: model. Keras model to train.
        worker_optimizer: string. String representing worker optimizer.
                          See: https://keras.io/optimizers/
        loss: string. String representing the loss function.
              See: https://keras.io/objectives/
        metrics: list of strings representing model evaluation metrics. Default is ["accuracy"].
                 See: https://keras.io/metrics/
        features_col: string or list of strings. Name(s) of the features column(s).
        num_epoch: int. Number of epochs.
        batch_size: int. Mini-batch size.
        num_workers: int. Number of distributed workers.
        communication_window: int. Staleness parameter.
                              This parameter describes the number of mini-batches that will be
                              computed before updating the center variable. For DOWNPOUR based
                              algorithms we recommend large communication windows.
    """

    def __init__(self, keras_model, worker_optimizer, loss, metrics=["accuracy"], num_workers=2, batch_size=32,
                 features_col="features", label_col="label", num_epoch=1, communication_window=12):
        # Initialize the parent object.
        super(ADAG, self).__init__(keras_model, worker_optimizer, loss, metrics, num_workers,
                                   batch_size, features_col, label_col, num_epoch)
        # Set algorithm parameters.
        self.communication_window = communication_window

    def allocate_worker(self):
        """Allocate an Adag worker."""
        worker = ADAGWorker(self.master_model, self.worker_optimizer, self.loss, self.metrics,
                            self.features_column, self.label_column, self.batch_size,
                            self.master_host, self.master_port, self.communication_window)

        return worker

    def allocate_parameter_server(self):
        """Allocate the Adag parameter server."""
        parameter_server = ADAGParameterServer(self.master_model, self.master_port)

        return parameter_server


class DynSGD(AsynchronousDistributedTrainer):
    """Dynamic SGD, dynamically maintains learning rate for every worker
    and incorperates staleness.

    Introduced in SIGMOD 2017 "Heterogenity-aware Parameter Servers"
    http://net.pku.edu.cn/~cuibin/Papers/2017SIGMOD.pdf

    # Arguments:
        keras_model: model. Keras model to train.
        worker_optimizer: string. String representing worker optimizer.
                          See: https://keras.io/optimizers/
        loss: string. String representing the loss function.
              See: https://keras.io/objectives/
        metrics: list of strings representing model evaluation metrics. Default is ["accuracy"].
                 See: https://keras.io/metrics/
        features_col: string or list of strings. Name(s) of the features column(s).
        num_epoch: int. Number of epochs.
        batch_size: int. Mini-batch size.
        num_workers: int. Number of distributed workers.
        communication_window: int. Staleness parameter.
                              This parameter describes the number of mini-batches that will be
                              computed before updating the center variable. For DOWNPOUR based
                              algorithms we recommend large communication windows.
    """

    def __init__(self, keras_model, worker_optimizer, loss, metrics=["accuracy"], num_workers=2, batch_size=32,
                 features_col="features", label_col="label", num_epoch=1, communication_window=5):
        # Initialize the parent object.
        super(DynSGD, self).__init__(keras_model, worker_optimizer, loss, metrics, num_workers,
                                     batch_size, features_col, label_col, num_epoch)
        # Set algorithm parameters.
        self.communication_window = communication_window

    def allocate_worker(self):
        """Allocate DYNSGD worker."""
        worker = DynSGDWorker(self.master_model, self.worker_optimizer, self.loss, self.metrics,
                              self.features_column, self.label_column, self.batch_size,
                              self.master_host, self.master_port, self.communication_window)

        return worker

    def allocate_parameter_server(self):
        """Allocate DYNSGD parameter server."""
        parameter_server = DynSGDParameterServer(self.master_model, self.master_port)

        return parameter_server


class Experimental(AsynchronousDistributedTrainer):
    """Experimental optimization scheme for development purposes."""

    def __init__(self, keras_model, worker_optimizer, loss, metrics=["accuracy"], num_workers=2, batch_size=32,
                 features_col="features", label_col="label", num_epoch=1, communication_window=5,
                 learning_rate=1.0):
        # Initialize the parent object.
        super(Experimental, self).__init__(keras_model, worker_optimizer, loss, metrics, num_workers,
                                           batch_size, features_col, label_col, num_epoch)
        # Set the algorithm parameters.
        self.communication_window = communication_window
        self.learning_rate = learning_rate

    def allocate_worker(self):
        """Allocate experimental worker."""
        worker = ExperimentalWorker(self.master_model, self.worker_optimizer, self.loss, self.metrics,
                                    self.features_column, self.label_column, self.batch_size,
                                    self.master_host, self.master_port, self.communication_window,
                                    self.num_workers, self.learning_rate)

        return worker

    def allocate_parameter_server(self):
        """Allocate experimental parameter server."""
        parameter_server = ExperimentalParameterServer(self.master_model, self.master_port, self.learning_rate)

        return parameter_server
