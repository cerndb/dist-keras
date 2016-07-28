import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

from distkeras.distributed_models import *

from pyspark import SparkContext, SparkConf


batch_size = 128
nb_classes = 10
nb_epoch = 20

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(),
              metrics=['accuracy'])

# Prepare Apache Spark
spark_config = SparkConf().setAppName("Dist-Keras Testing").setMaster('local[*]')
spark_context = SparkContext(conf=spark_config)

rdd = to_simple_rdd(spark_context, X_train, Y_train)
sparkModel = SparkModel(spark_context, rdd, keras_model=model, optimizer=SGD(), loss='categorical_crossentropy', num_workers=4)

parameters = {}
parameters['nb_epoch'] = nb_epoch
parameters['batch_size'] = batch_size
sparkModel.train(parameters)
sparkModel.stop_server()

score = sparkModel.master_model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
