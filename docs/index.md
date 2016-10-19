# Distributed Keras

Distributed Keras (DK) is a **distributed deep learning framework** built op top of Apache Spark and Keras with the goal to significantly reduce the training time using distributed machine learning algorithms. We designed the framework in such a way that a developer could implement a new distributed optimizer with ease, thus enabling a person to focus on research and model development.

As mentioned above, most of our methods follow the data parallel approach as described in the paper on [Large Scale Distributed Deep Networks](http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf). In this paradigm, replicas of a model are distributed over several "trainers", and every model replica will be trained on a different partition of the dataset. The gradient (or all network weights, depending on the implementation details) will be communicated with the parameter server after every gradient update. The parameter server is responsible for handling the gradient updates of all workers and incorperating all gradient updates into a single master model which will be returned to the user after the training procedure is complete.

## Installation

We rely on [Keras](https://keras.io) for the construction of models, and thus following the Keras dependencies. Furthermore, PySpark is also a dependency for this project since DK is using Apache Spark for the distribution of the data and the model replicas.

### Pip

You can use `pip` if you only need to DK framework without examples.

```bash
pip install git+https://github.com/JoeriHermans/dist-keras.git
```

### Git

However, if you would like to play with the examples and notebooks, simply install the framework using the approach described below.

```bash
git clone https://github.com/JoeriHermans/dist-keras
cd dist-keras
pip install -e .
```

## Getting Started

We recommend starting with the `workflow` notebook located in the `examples` directory. This Python notebook will guide you through all general steps which should need to perform. This includes setting up a Spark Context, reading the data, applying preprocessing, training and evaluation of your model in a distributed way.

!!! Note
    Running the **workflow.ipyn** notebook can be run on your local machine. However, we recommend running the notebook on a Spark cluster since the distributed trainers start to outperform the *SingleTrainer* when the number of workers (cores multiplied by executors) is usually higher than 10.

## Support

For issues, bugs, questions, and suggestions. Please use the appropriate channels on [GitHub](https://github.com/JoeriHermans/dist-keras/).

After the installation process is complete, you can start exploring the functionality by browsing the examples. We have also prepared a notebook which basically compares the different distributed optimizers with each other. This notebook is located at `examples/experiment.ipynb`. However, other examples are also provided which show you how to use the different distributed optimizers with Apache Spark for distributed pre-processing.

## References

* Zhang, S., Choromanska, A. E., & LeCun, Y. (2015). Deep learning with elastic averaging SGD. In Advances in Neural Information Processing Systems (pp. 685-693).

* Moritz, P., Nishihara, R., Stoica, I., & Jordan, M. I. (2015). SparkNet: Training Deep Networks in Spark. arXiv preprint arXiv:1511.06051.

* Dean, J., Corrado, G., Monga, R., Chen, K., Devin, M., Mao, M., ... & Ng, A. Y. (2012). Large scale distributed deep networks. In Advances in neural information processing systems (pp. 1223-1231).

* Pumperla, M. (2015). Elephas. Github Repository https://github.com/maxpumperla/elephas/. [4]

## Licensing

![GPLv3](images/gpl_v3.png) ![CERN](images/cern_logo.jpg)
