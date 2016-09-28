# Distributed Keras

Distributed Deep Learning with Apache Spark and Keras.

Attention: since this is alpha software, I have hardcoded the loss in the workers. You can change this easily by modifing the compilation arguments.

## Introduction

## Algorithms

### SingleTrainer

```python
SingleTrainer(keras_model, num_epoch=1, batch_size=1000, features_col="features", label_col="label")
```

### Ensemble Trainer

```python
EnsembleTrainer(keras_model, num_models=2, merge_models=False, features_col="features", label_col="label")
```

### EASGD

```python
EASGD(keras_model, num_workers=2, rho=5.0, learning_rate=0.01, batch_size=1000, features_col="features", label_col="label")
```

### DPGO (Experimental)

```python
DPGO(keras_model, num_workers=2, batch_size=1000, features_col="features", label_col="label")
```

## Utility classes

### Transformers

### Predictors

## References

* Zhang, S., Choromanska, A. E., & LeCun, Y. (2015). Deep learning with elastic averaging SGD. In Advances in Neural Information Processing Systems (pp. 685-693).

## Licensing

![GPLv3](resources/gpl_v3.png)
