from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

from distkeras.distributed_models import *
from distkeras import optimizers as distkeras_optimizers

from pyspark import SparkContext, SparkConf

batch_size = 64
nb_classes = 10
nb_epoch = 3

conf = SparkConf().setAppName("Dist-Keras Testing").setMaster('local[*]')
sc = SparkContext(conf=conf)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(128, input_dim=784))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))
sgd = SGD(lr=0.1)
loss='categorical_crossentropy'
model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])
rdd = to_simple_rdd(sc, x_train, y_train)
sgd = distkeras_optimizers.SGD()
sparkModel = SparkModel(sc, rdd, keras_model=model, optimizer=sgd, loss=loss)

parameters = {}
parameters['nb_epoch'] = nb_epoch
parameters['batch_size'] = batch_size
sparkModel.train(parameters)

score = sparkModel.master_model.evaluate(x_test, y_test, batch_size=64, verbose=1)
print(sparkModel.master_model.metrics_names)
print(score)
