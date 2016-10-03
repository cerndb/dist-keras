# Distributed Keras

Distributed Deep Learning with Apache Spark and Keras.

**Attention**: since this is alpha software, I have hardcoded the loss in the workers. You can change this easily by modifing the compilation arguments.

## Introduction

Distributed Keras is a distributed deep learning framework built on top of Apache Spark and Keras. We designed the framework in such a way that a developer could implement a new distributed optimizer with ease, thus enabling a person to focus on research. Several distributed methods are supported, such as, but not restricted to, the training of **ensemble models**, and **data parallel** models.

As discussed above, most methods are implemented as data parallel models. Data parallel models, as described in [[3]](http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf), is a learning paradigm where multiple replicas of a model are used to optimize a single objective. Using data parallelism, we are able to significantly increase the training time of the model. Depending on the parametrization, we are able to achieve better model performance compared to a more common training approach (e.g., like the [SingleTrainer](#single-trainer) implementation), and yet, spending less time on the training of the model.

## Installation

We will guide you how to install Distributed Keras. However, we will assume that an Apache Spark installation is available.

### pip

When you only require the framework, just use `pip` to install dist-keras.

```bash
pip install git+https://github.com/JoeriHermans/dist-keras.git
```
### git

Using this approach, you will be able to easily execute the examples.

```bash
git clone https://github.com/JoeriHermans/dist-keras
```

In order to install possible missing dependencies, and to compile the dist-keras modules, we need to run `pip`.

```bash
cd dist-keras
pip install -e .
```

### General notes

#### .bashrc

Make sure the following variables are set in your `.bashrc`. It is possible, depending on your system configuration, that the following configuration **doesn't have to be applied**.

```bash
# Example of a .bashrc configuration.
export SPARK_HOME=/usr/lib/spark
export PYTHONPATH="$SPARK_HOME/python/:$SPARK_HOME/python/lib/py4j-0.9-src.zip:$PYTHONPATH"
```

## Running an example

TODO

### Spark 2.0

If you want to run the examples using Apache Spark 2.0.0 and higher. You will need to remove the line containing `sqlContext = SQLContext(sc)`. We need to do this because in Spark 2.0+, the SQLContext, and Hive context are now merged in the Spark session.

## Algorithms

### Single Trainer

A single trainer is in all simplicity a trainer which will use a single Spark executor to train a model. This trainer is usually used as a baseline metrics for new distributed optimizers.

```python
SingleTrainer(keras_model, num_epoch=1, batch_size=1000, features_col="features", label_col="label")
```

### Ensemble Trainer

This trainer will employ [ensemble learning](https://en.wikipedia.org/wiki/Ensemble_learning) to build a classifier. There are two modes, in the first mode, you will get a list of Keras models which have been trained on different partitions of the data. In the other mode, all the models will be merged by adding an averaging layer to the networks. The resulting model will thus have the same outputs as the specified Keras model, with the difference that the actual output is the averaged output of the parallely trained models.

```python
EnsembleTrainer(keras_model, num_models=2, merge_models=False, features_col="features", label_col="label")
```

### EASGD

The distinctive idea of EASGD is to allow the local workers to perform more exploration (small rho) and the master to perform exploitation. This approach differs from other settings explored in the literature, and focus on how fast the center variable converges [[1]](https://arxiv.org/pdf/1412.6651.pdf) .

We want to note the basic version of EASGD is a synchronous algorithm, i.e., once a worker is done processing a batch of the data, it will wait until all other workers have submitted their variables (this includes the weight parameterization, iteration number, and worker id) to the parameter server before starting the next data batch.

```python
EASGD(keras_model, num_workers=2, rho=5.0, learning_rate=0.01, batch_size=1000, features_col="features", label_col="label")
```

### Asynchronous EASGD

TODO

```python
AsynchronousEASGD(keras_model, num_workers=2, rho=5.0, learning_rate=0.01, batch_size=1000, features_col="features", label_col="label", communcation_window=5)
```

### DOWNPOUR

TODO

```python
DOWNPOUR(keras_model, learning_rate=0.01, num_workers=2, batch_size=1000, features_col="features", label_col="label", communication_window=5)
```

## Utility classes

### Transformers

TODO

### Predictors

TODO

## References

* Zhang, S., Choromanska, A. E., & LeCun, Y. (2015). Deep learning with elastic averaging SGD. In Advances in Neural Information Processing Systems (pp. 685-693). [1]

* Moritz, P., Nishihara, R., Stoica, I., & Jordan, M. I. (2015). SparkNet: Training Deep Networks in Spark. arXiv preprint arXiv:1511.06051. [2]

* Dean, J., Corrado, G., Monga, R., Chen, K., Devin, M., Mao, M., ... & Ng, A. Y. (2012). Large scale distributed deep networks. In Advances in neural information processing systems (pp. 1223-1231). [3]

## Licensing

![GPLv3](resources/gpl_v3.png) ![CERN](resources/cern_logo.jpg)
