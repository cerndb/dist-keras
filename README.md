[![DOI](https://zenodo.org/badge/64122944.svg)](https://zenodo.org/badge/latestdoi/64122944)

# Distributed Keras

Distributed Deep Learning with Apache Spark and Keras.


## Introduction

Distributed Keras is a distributed deep learning framework built op top of Apache Spark and Keras. We designed the framework in such a way that a new distributed optimizer could be implemented with ease, thus enabling a person to focus on research. Several distributed methods are supported, such as, but not restricted to, the training of **ensembles** and models using **data parallel** methods.

Most of the distributed optimizers we provide, are based on data parallel methods. A data parallel method, as described in [[1]](http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf), is a learning paradigm where multiple replicas of a single model are used to optimize a single objective. Using this approach, we are able to dignificantly reduce the training time of a model. Depending on the parametrization, we also observed that it is possible to achieve better statistical model performance compared to a more traditional approach (e.g., like the [SingleTrainer](#single-trainer) implementation), and yet, spending less wallclock time on the training of the model. However, this is subject to further research.

**Attention**: We recommend reading the [workflow](https://github.com/JoeriHermans/dist-keras/blob/master/examples/workflow.ipynb) Jupyter notebook. This includes a complete description of the problem, how to use it, preprocess your data with Apache Spark, and a performance evaluation of all included distributed optimizers.


## Installation

We will guide you how to install Distributed Keras. However, we will assume that an Apache Spark installation is available. In the following subsections, we describe two approaches to achieve this.

### pip

When you only require the framework for development purposes, just use `pip` to install dist-keras.

```bash
pip install --upgrade git+https://github.com/JoeriHermans/dist-keras.git
```

### git & pip

However, if you would like to contribute, or run some of the examples. It is probably best to clone the repository directly from GitHub and install it afterwards using `pip`. This will also resolve possible missing dependencies.

```bash
git clone https://github.com/JoeriHermans/dist-keras
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

We would like to refer the reader to the `workflow.ipynb` notebook in the examples folder. This will give you a complete introduction to the problem of distributed deep learning, and will guide you through the steps that have to be executed.

Furthermore, we would also like to show how you exactly should process "big" datasets. This is shown in the examples starting with the prefix ```example_```. Please execute them in the provided sequence.

### Spark 2.0

If you want to run the examples using Apache Spark 2.0.0 and higher. You will need to remove the line containing `sqlContext = SQLContext(sc)`. We need to do this because in Spark 2.0+, the SQLContext, and Hive context are now merged in the Spark session.


## Algorithms

### Single Trainer

This optimizer follows the traditional scheme of training a model, i.e., it uses sequential gradient updates to optimize the parameters. It does this by executing the training procedure on a single Spark executor.

```python
SingleTrainerWorker(model, features_col, label_col, batch_size, optimizer, loss)
```

### Asynchronous Elastic Averaging SGD (AEASGD)

The distinctive idea of EASGD is to allow the local workers to perform more exploration (small rho) and the master to perform exploitation. This approach differs from other settings explored in the literature, and focus on how fast the center variable converges [[2]](https://arxiv.org/pdf/1412.6651.pdf) .

In this section we show the asynchronous version of EASGD. Instead of waiting on the synchronization of other trainers, this method communicates the elastic difference (as described in the paper), with the parameter server. The only synchronization mechanism that has been implemented, is to ensure no race-conditions occur when updating the center variable.


```python
AEASGD(keras_model, worker_optimizer, loss, num_workers, batch_size, features_col,
       label_col, num_epoch, communication_window, rho, learning_rate)
```

### Asynchronous Elastic Averaging Momentum SGD (AEAMSGD)

Asynchronous EAMSGD is a variant of asynchronous EASGD. It is based on the Nesterov's momentum scheme, where the update of the local worker is modified to incorepare a momentum term [[2]](https://arxiv.org/pdf/1412.6651.pdf) .

```python
EAMSGD(keras_model, worker_optimizer, loss, num_workers, batch_size,
       features_col, label_col, num_epoch, communication_window, rho,
       learning_rate, momentum)
```

### DOWNPOUR

An asynchronous stochastic gradient descent procedure introduced by Dean et al., supporting a large number of model replicas and leverages adaptive learning rates. This implementation is based on the pseudocode provided by [[1]](http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf) .

```python
DOWNPOUR(keras_model, worker_optimizer, loss, num_workers, batch_size,
         features_col, label_col, num_epoch, learning_rate, communication_window)
```

### Ensemble Training

In ensemble training, we train `n` models in parallel on the same dataset. All models are trained in parallel, but the training of a single model is done in a sequential manner using Keras optimizers. After the training process, one can combine and, for example, average the output of the models.

```python
EnsembleTrainer(keras_model, worker_optimizer, loss, features_col,
                label_col, batch_size, num_ensembles)
```

### Model Averaging

Model averaging is a data parallel technique which will average the trainable parameters of model replicas after every epoch.

```python
AveragingTrainer(keras_model, worker_optimizer, loss, features_col,
                 label_col, num_epoch, batch_size, num_workers)
```

## General note

It is known that adding more asynchronous workers deteriorates the statistical performance of the model. There have been some studies which examinate this particular effect. However, some of them conclude that actually adding more asynchronous workers contributes to something what they call **implicit momentum** [[3]](https://arxiv.org/pdf/1605.09774.pdf). However, this is subject to further investigation.


## Known issues

- Python 3 compatibility.


## TODO's

List of possible future additions.

- Save Keras model to HDFS.
- Load Keras model from HDFS.
- Compression / decompression of network transmissions.
- Monitoring of loss and training.
- Stop on target loss.
- Multiple parameter servers for large Deep Networks.
- Python 3 compatibility.
- For every worker, spawn an additional thread which is responsible for sending updates to the parameter server. The actual worker thread will just submit tasks to this queue.


## Citing

If you use this framework in any academic work, please use the following BibTex code.

```latex
@misc{dist_keras_joerihermans,
  author = {Joeri Hermans and CERN IT-DB},
  title = {Distributed Keras: Distributed Deep Learning with Apache Spark and Keras},
  year = {2016},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/JoeriHermans/dist-keras/}},
}
```

## Citing

If you use this software in your work, please cite us.

```
@misc{distributed_keras,
  author       = {Joeri Hermans},
  title        = {Distributed Keras: Distributed Deep Learning with Apache Spark and Keras},
  month        = nov,
  year         = 2016,
  doi          = {10.5281/zenodo.167573},
  url          = {https://doi.org/10.5281/zenodo.167573}
}
```

## References

* Dean, J., Corrado, G., Monga, R., Chen, K., Devin, M., Mao, M., ... & Ng, A. Y. (2012). Large scale distributed deep networks. In Advances in neural information processing systems (pp. 1223-1231). [[1]](http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf)

* Zhang, S., Choromanska, A. E., & LeCun, Y. (2015). Deep learning with elastic averaging SGD. In Advances in Neural Information Processing Systems (pp. 685-693). [[2]](https://arxiv.org/pdf/1412.6651.pdf)

* Mitliagkas, Ioannis, et al. "Asynchrony begets Momentum, with an Application to Deep Learning." arXiv preprint arXiv:1605.09774 (2016). [[3]](https://arxiv.org/pdf/1605.09774.pdf)

<!-- @misc{pumperla2015, -->
<!-- author = {Max Pumperla}, -->
<!-- title = {elephas}, -->
<!-- year = {2015}, -->
<!-- publisher = {GitHub}, -->
<!-- journal = {GitHub repository}, -->
<!-- howpublished = {\url{https://github.com/maxpumperla/elephas}} -->
<!-- } -->
* Pumperla, M. (2015). Elephas. Github Repository https://github.com/maxpumperla/elephas/. [4]


## Licensing

![GPLv3](resources/gpl_v3.png) ![CERN](resources/cern_logo.jpg)
