import numpy as np

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical

from pyspark import SparkContext
from pyspark import SparkConf
from pyspark import SQLContext

from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.sql import Row
from pyspark.mllib.linalg import DenseVector

from itertools import tee
from itertools import chain

import os

# Setup Spark, and use the Databricks CSV loader.
os.environ['PYSPARK_SUBMIT_ARGS'] = "--packages com.databricks:spark-csv_2.10:1.4.0 pyspark-shell"
# Setup the Spark -, and SQL Context (note: this is for Spark < 2.0.0)
sc = SparkContext(appName="DistKeras ATLAS Higgs example")
sqlContext = SQLContext(sc)

# Read the Higgs dataset.
dataset = sqlContext.read.format('com.databricks.spark.csv')\
                    .options(header='true', inferSchema='true').load("data/atlas_higgs.csv");
# Print the schema of the dataset.
dataset.printSchema()
# Vectorize the features into the features column.
features = dataset.columns
features.remove('EventId')
features.remove('Weight')
features.remove('Label')
assembler = VectorAssembler(inputCols=features, outputCol="features")
dataset = assembler.transform(dataset)
# Since the output layer will not be able to read the string label, convert it to an double.
labelIndexer = StringIndexer(inputCol="Label", outputCol="label").fit(dataset)
dataset = labelIndexer.transform(dataset)
# Feature normalization.
standardScaler = StandardScaler(inputCol="features", outputCol="features_normalized", withStd=True, withMean=True)
standardScalerModel = standardScaler.fit(dataset)
dataset = standardScalerModel.transform(dataset)
# Select the required columns.
dataset = dataset.select(["features_normalized", "label"])
# Split the dataset in a training and testset.
(trainingSet, testSet) = dataset.randomSplit([0.7, 0.3])

# Define the structure of the dataset.
nb_features = len(features)
nb_classes = 2

# Define the Keras model.
model = Sequential()
model.add(Dense(600, input_shape=(nb_features,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(600))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(600))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# Print a summary of the model structure.
model.summary()

jsonModel = model.to_json()

def to_vector(x):
    vector = np.zeros(2)
    vector[x] = 1

    return vector

def method_train(iterator):
    global jsonModel

    # Allocate empty feature and label arrays.
    feature_iterator, label_iterator = tee(iterator, 2)
    X = np.asarray([x.features_normalized for x in feature_iterator])
    y = np.asarray([to_vector(x.label) for x in label_iterator])
    model = model_from_json(jsonModel)
    # Compile the Keras model.
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    # Fit the model with the data.
    history = model.fit(X, y, nb_epoch=1)
    partitionResult = (history, model.to_json())
    # Set the new global JSON model.
    jsonModel = model.to_json()

    return iter([partitionResult])

def method_predict(iterator):
    global jsonModel

    # Deserialize the JSON Keras model
    model = model_from_json(jsonModel)
    rows = []
    try:
        for row in iterator:
            X = np.asarray([row.features_normalized])
            Y = model.predict(X)
            densePredictionVector = DenseVector(Y.tolist())
            new_row = Row(row.__fields__ + ['prediction'])(row + (densePredictionVector,))
            rows.append(new_row)
    except TypeError:
        pass

    return iter(rows)

dataset = dataset.repartition(1).sample(True, 0.001)
models = dataset.mapPartitions(method_train).collect()

# Load the validation set.
validationset = sqlContext.read.format('com.databricks.spark.csv')\
                          .options(header='true', inferSchema='true').load("data/atlas_higgs_test.csv");
# Apply the same steps as with the original dataset.
validationset = assembler.transform(validationset).sample(True, 0.01)
standardScalerModel = standardScaler.fit(validationset)
validationset = standardScalerModel.transform(validationset).sample(True, 0.01)

validationset.printSchema()
validationset = validationset.mapPartitions(method_predict)
set = validationset.collect()
print(set[0])
validationset = validationset.toDF()
validationset.printSchema()
