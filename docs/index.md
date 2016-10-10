# Distributed Keras

Distributed Keras (DK) is a **distributed deep learning framework** built op top of Apache Spark and Keras with the goal to significantly reduce the training time using distributed machine learning algorithms. We designed the framework in such a way that a developer could implement a new distributed optimizer with ease, thus enabling a person to focus on research and model development. Several distributed methods are implemented, such as, but not restricted to the training of **ensemble models**, and **data parallel** models.

As mentioned above, most of our methods follow the data parallel approach as described in the paper on [Large Scale Distributed Deep Networks](http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf). In this paradigm, replicas of a model are distributed over several "trainers", and every model replica will be trained on a different partition of the dataset. The gradient (or all network weights, depending on the implementation details) will be communicated with the parameter server after every gradient update. The parameter server is responsible for handling the gradient updates of all workers and incorperating all gradient updates into a single master model which will be returned to the user after the training procedure is complete.

!!! warning
    Since this is alpha software, we have hardcoded the loss in the workers for now. You can change this easily by modifying the compilation arguments of the models.

## Installation

We rely on [Keras](https://keras.io) for the construction of models, and thus following the Keras dependencies. Furthermore, PySpark is also a dependency for this project since DK is using Apache Spark for the distribution of the data and the model replicas.

### Pip

You can use `pip` if you only need to DK framework without examples.

```bash
pip install git+https://github.com/JoeriHermans/dist-keras.git
```

### Git

However, if you would like to play with the examples and install the framework, it is recommended to use to following approach.

```bash
git clone https://github.com/JoeriHermans/dist-keras
cd dist-keras
pip install -e .
```

## Architecture

## Getting Started

## References

* Zhang, S., Choromanska, A. E., & LeCun, Y. (2015). Deep learning with elastic averaging SGD. In Advances in Neural Information Processing Systems (pp. 685-693).

* Moritz, P., Nishihara, R., Stoica, I., & Jordan, M. I. (2015). SparkNet: Training Deep Networks in Spark. arXiv preprint arXiv:1511.06051.

* Dean, J., Corrado, G., Monga, R., Chen, K., Devin, M., Mao, M., ... & Ng, A. Y. (2012). Large scale distributed deep networks. In Advances in neural information processing systems (pp. 1223-1231).

## Licensing

![GPLv3](images/gpl_v3.png) ![CERN](images/cern_logo.jpg)
