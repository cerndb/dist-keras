# Distributed Keras

Distributed Deep Learning with Apache Spark and Keras.


## Introduction

Distributed Keras is a distributed deep learning framework built op top of Apache Spark and Keras, with a focus on "state-of-the-art" distributed optimization algorithms. We designed the framework in such a way that a new distributed optimizer could be implemented with ease, thus enabling a person to focus on research. Several distributed methods are supported, such as, but not restricted to, the training of **ensembles** and models using **data parallel** methods.

Most of the distributed optimizers we provide, are based on data parallel methods. A data parallel method, as described in [[1]](http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf), is a learning paradigm where multiple replicas of a single model are used to optimize a single objective. Using this approach, we are able to dignificantly reduce the training time of a model. Depending on the parametrization, we also observed that it is possible to achieve better statistical model performance compared to a more traditional approach (e.g., like the [SingleTrainer](#single-trainer) implementation), and yet, spending less wallclock time on the training of the model. However, this is subject to further research.

**Attention**: A rather complete introduction to the problem of Distributed Deep Learning is presented in my Master Thesis [http://github.com/JoeriHermans/master-thesis](http://github.com/JoeriHermans/master-thesis). Furthermore, the thesis describes includes several *novel* insights, such as a redefinition of parameter staleness, and several new distributed optimizers such as AGN and ADAG.


## Installation

We will guide you how to install Distributed Keras. However, we will assume that an Apache Spark installation is available. In the following subsections, we describe two approaches to achieve this.

### pip

When you only require the framework for development purposes, just use `pip` to install dist-keras.

```bash
pip install --upgrade dist-keras

# OR

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


## Optimization Algorithms

### Sequential Trainer

This optimizer follows the traditional scheme of training a model, i.e., it uses sequential gradient updates to optimize the parameters. It does this by executing the training procedure on a single Spark executor.

```python
SingleTrainer(model, features_col, label_col, batch_size, optimizer, loss, metrics=["accuracy"])
```

### ADAG (Currently Recommended)

DOWNPOUR variant which is able to achieve significantly better statistical performance while being less sensitive to hyperparameters. This optimizer was developed using insights gained while developing this framework. More research regarding parameter staleness is still being conducted to further improve this optimizer.

```python
ADAG(keras_model, worker_optimizer, loss, metrics=["accuracy"], num_workers=2, batch_size=32,
     features_col="features", label_col="label", num_epoch=1, communication_window=12)
```

### Dynamic SGD

Dynamic SGD, dynamically maintains a learning rate for every worker by incorperating parameter staleness. This optimization scheme is introduced in "Heterogeneity-aware Distributed Parameter Servers" at the SIGMOD 2017 conference [[5]](http://net.pku.edu.cn/~cuibin/Papers/2017SIGMOD.pdf).

```python
DynSGD(keras_model, worker_optimizer, loss, metrics=["accuracy"], num_workers=2, batch_size=32,
       features_col="features", label_col="label", num_epoch=1, communication_window=10)
```

### Asynchronous Elastic Averaging SGD (AEASGD)

The distinctive idea of EASGD is to allow the local workers to perform more exploration (small rho) and the master to perform exploitation. This approach differs from other settings explored in the literature, and focus on how fast the center variable converges [[2]](https://arxiv.org/pdf/1412.6651.pdf) .

In this section we show the asynchronous version of EASGD. Instead of waiting on the synchronization of other trainers, this method communicates the elastic difference (as described in the paper), with the parameter server. The only synchronization mechanism that has been implemented, is to ensure no race-conditions occur when updating the center variable.


```python
AEASGD(keras_model, worker_optimizer, loss, metrics=["accuracy"], num_workers, batch_size, features_col,
       label_col, num_epoch, communication_window, rho, learning_rate)
```

### Asynchronous Elastic Averaging Momentum SGD (AEAMSGD)

Asynchronous EAMSGD is a variant of asynchronous EASGD. It is based on the Nesterov's momentum scheme, where the update of the local worker is modified to incorepare a momentum term [[2]](https://arxiv.org/pdf/1412.6651.pdf) .

```python
EAMSGD(keras_model, worker_optimizer, loss, metrics=["accuracy"], num_workers, batch_size,
       features_col, label_col, num_epoch, communication_window, rho,
       learning_rate, momentum)
```

### DOWNPOUR

An asynchronous stochastic gradient descent procedure introduced by Dean et al., supporting a large number of model replicas and leverages adaptive learning rates. This implementation is based on the pseudocode provided by [[1]](http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf) .

```python
DOWNPOUR(keras_model, worker_optimizer, loss, metrics=["accuracy"], num_workers, batch_size,
         features_col, label_col, num_epoch, learning_rate, communication_window)
```

### Ensemble Training

In ensemble training, we train `n` models in parallel on the same dataset. All models are trained in parallel, but the training of a single model is done in a sequential manner using Keras optimizers. After the training process, one can combine and, for example, average the output of the models.

```python
EnsembleTrainer(keras_model, worker_optimizer, loss, metrics=["accuracy"], features_col,
                label_col, batch_size, num_ensembles)
```

### Model Averaging

Model averaging is a data parallel technique which will average the trainable parameters of model replicas after every epoch.

```python
AveragingTrainer(keras_model, worker_optimizer, loss, metrics=["accuracy"], features_col,
                 label_col, num_epoch, batch_size, num_workers)
```

## Job deployment

We also support remote job deployment. For example, imagine you are developing your model on a local notebook using a small development set. However, in order to submit your job on a remote cluster, you first need to develop a cluster job, and run the job there. In order to simplify this process, we have developed a simplified interface for a large scale machine learning job.

In order to submit a job to a remote cluster, you simply run the following code:

```python
# Define the distributed optimization procedure, and its parameters.
trainer = ADAG(keras_model=mlp, worker_optimizer=optimizer_mlp, loss=loss_mlp, metrics=["accuracy"], num_workers=20,
               batch_size=32, communication_window=15, num_epoch=1,
               features_col="features_normalized_dense", label_col="label_encoded")

# Define the job parameters.
job = Job(secret, job_name, data_path, num_executors, num_processes, trainer)
job.send('http://yourcluster:[port]')
job.wait_completion()
# Fetch the trained model, and history for training evaluation.
trained_model = job.get_trained_model()
history = job.get_history()
```

### Punchcard Server

Job scheduling, and execution is handled by our `Punchcard` server. This server will accept requests from a remote location given a specific `secret`, which is basically a long identification string of a specific user. However, a user can have multiple secrets. At the moment, a job is only executed if there are no other jobs running for the specified secret.

In order to submit jobs to `Punchcard` we need to specify a secrets file. This file is basically a JSON structure, it will have the following structure:

```json
[
    {
        "secret": "secret_of_user_1",
        "identity": "user1"
    },
    {
        "secret": "secret_of_user_2",
        "identity": "user2"
    }
]
```

After the secrets file has been constructed, the Punchcard server can be started by issueing the following command.

```sh
python scripts/punchcard.py --secrets /path/to/secrets.json
```

#### Secret Generation

In order to simplify secret generation, we have added a costum script which will generate a unique key for the specified identity. The structure can be generated by running the following command.

```sh
python scripts/generate_secret.py --identity userX
```

## Optimization Schemes

TODO

## General note

It is known that adding more asynchronous workers deteriorates the statistical performance of the model. There have been some studies which examinate this particular effect. However, some of them conclude that actually adding more asynchronous workers contributes to something what they call **implicit momentum** [[3]](https://arxiv.org/pdf/1605.09774.pdf). However, this is subject to further investigation.


## Known issues

- Python 3 compatibility.


## TODO's

List of possible future additions.

- Save Keras model to HDFS.
- Load Keras model from HDFS.
- Compression / decompression of network transmissions.
- Stop on target loss.
- Multiple parameter servers for large Deep Networks.
- Python 3 compatibility.
- For every worker, spawn an additional thread which is responsible for sending updates to the parameter server. The actual worker thread will just submit tasks to this queue.


## Citing

If you use this framework in any academic work, please use the following BibTex code.

```latex
@misc{dist_keras_joerihermans,
  author = {Joeri R. Hermans, CERN IT-DB},
  title = {Distributed Keras: Distributed Deep Learning with Apache Spark and Keras},
  year = {2016},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/JoeriHermans/dist-keras/}},
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
* Jiawei Jiang, Bin Cui, Ce Zhang and Lele Yu (2017). Heterogeneity-aware Distributed Parameter Servers [[5]](http://net.pku.edu.cn/~cuibin/Papers/2017SIGMOD.pdf)


## Licensing

![GPLv3](resources/gpl_v3.png) ![CERN](resources/cern_logo.jpg)
