# Optimizers

Optimizers, or trainers, are the main component in Distributed Keras (DK). All trainers share a single interface, which is the `Trainer` class, defined in `distkeras/distributed.py`. This class also contains the `serialized model`, the `loss`, and the `Keras optimizer` the workers need to use. Generally, a trainer will run on a single worker. In the context of Apache Spark, this means that the thread which is responsible for doing the foreachPartition or mapPartitions will have been assigned a trainer. In reality however, the training of the model itself will utilise more physical cores. In fact, it will employ all available cores, and thus bypassing resource managers such as YARN.

## Single Trainer

A single trainer is in all simplicity a trainer which will use a single thread (as discussed above) to train a model. This trainer is usually used as a baseline metric for new distributed optimizers.

```python
SingleTrainer(keras_model, worker_optimizer, loss, metrics=["accuracy"], num_epoch=1,
              batch_size=32, features_col="features", label_col="label")
```
**Parameters**:

- **keras_model**:            The Keras model which should be trained.
- **worker_optmizer**:        Keras optimizer for workers.
- **num_epoch**:              Number of epoch iterations over the data.
- **batch_size**:             Mini-batch size.
- **features_col**:           Column of the feature vector in the Spark Dataframe.
- **label_col**:              Column of the label in the Spark Dataframe.

## EASGD

The distinctive idea of EASGD is to allow the local workers to perform more exploration (small rho) and the master to perform exploitation. This approach differs from other settings explored in the literature, and focus on how fast the center variable converges [(paper)](https://arxiv.org/pdf/1412.6651.pdf) .

We want to note the basic version of EASGD is a synchronous algorithm, i.e., once a worker is done processing a batch of the data, it will wait until all other workers have submitted their variables (this includes the weight parameterization, iteration number, and worker id) to the parameter server before starting the next data batch.

```python
EASGD(keras_model, worker_optimizer, loss, metrics=["accuracy"], num_workers=2, 
      features_col="features", label_col="label", rho=5.0, learning_rate=0.01, 
      batch_size=32, num_epoch=1, master_port=5000)
```

**Parameters**:

TODO

## Asynchronous EASGD

In this section we propose the asynchronous version of EASGD. Instead of waiting on the synchronization of other trainers, this method communicates the elastic difference (as described in the paper), with the parameter server. The only synchronization mechanism that has been implemented, is to ensure no race-conditions occur when updating the center variable.

```python
AsynchronousEASGD(keras_model, worker_optimizer, loss, metrics=["accuracy"], num_workers=2, batch_size=1000,
                  features_col="features", label_col="label", communication_window=3,
                  rho=0.01, learning_rate=0.01, master_port=5000, num_epoch=1)
```

**Parameters**:

TODO

## Asynchronous EAMSGD

Asynchronous EAMSGD is a variant of asynchronous EASGD. It is based on the Nesterov's momentum scheme, where the update of the local worker is modified to incorepare a momentum term.

```python
AsynchornousEAMSGD(keras_model, worker_optimizer, loss, metrics=["accuracy"], num_workers=2, batch_size=32,
                  features_col="features", label_col="label", communication_window=10,
                  rho=5.0, learning_rate=0.01, momentum=0.9, master_port=5000, num_epoch=1)
```

**Parameters**:

TODO

## DOWNPOUR

An asynchronous stochastic gradient descent procedure supporting a large number of model replicas and leverages adaptive learning rates. This implementation is based on the pseudocode provided by [Zhang et al.](https://arxiv.org/pdf/1412.6651.pdf)

```python
DOWNPOUR(keras_model, worker_optimizer, loss, metrics=["accuracy"], num_workers=2, batch_size=1000,
         features_col="features", label_col="label", communication_window=5,
         master_port=5000, num_epoch=1, learning_rate=0.01))
```

**Parameters**:

TODO

## Custom distributed optimizer

TODO

### Synchronized Distributed Trainer

TODO

### Asynchronous Distributed Trainer

TODO

### Implementing a custom worker

TODO
