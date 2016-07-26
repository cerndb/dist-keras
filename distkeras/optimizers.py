"""
Essentially a copy of Keras optimizers. However, the methods will only be used
as a way to get the configuration and to compute the updates without actually
storing them.
"""

## BEGIN Imports. ##############################################################

from keras import backend as K
from keras.utils.generic_utils import get_from_module

from six.moves import zip

import numpy as np

## END Imports. ################################################################

## BEGIN Utility functions. ####################################################

def clip_norm(g, c, n):
    if c > 0:
        g = K.switch(K.ge(n, c), g * c / n, g)

    return g

def kl_divergence(p, p_hat):
    return p_hat - p + p * K.log(p / p_hat)

## END Utility functions. ######################################################

class Optimizer(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get_updates(self, params, constraints, grads):
        raise NotImplementedError

    def get_config(self):
        return {"name": self.__class__.__name__}

class SGD(Optimizer):

    def __init__(self, lr=0.01, momentum=0, decay=0,
                 nesterov=False, *args, **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = 0
        self.lr = 0
        self.nesterov = nesterov
        self.momentum = momentum
        self.decay = decay

    def get_updates(self, params, constraints, grads):
        lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))
        self.iterations += 1
        new_weights = []

        for p, g, c in zip(params, grads, constraints):
            m = np.zeros_like(p)
            v = self.momentum * m - lr * g
            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v
            new_weights.append(c(new_p))

        return new_weights

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(self.lr),
                "momentum": float(self.momentum),
                "decay": float(self.decay),
                "nesterov": self.nesterov}
