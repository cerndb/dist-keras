import numpy as np

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical

from flask import Flask, request

from multiprocessing import Process, Lock

import cPickle as pickle

import urllib2

import time

from pyspark import SparkContext
from pyspark import SparkConf
from pyspark import SQLContext

from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from distkeras.distributed import EnsembleTrainer
from distkeras.distributed import LabelVectorTransformer
from distkeras.distributed import ModelPredictor
from distkeras.distributed import LabelIndexTransformer
from distkeras.distributed import Trainer
from distkeras.distributed import new_dataframe_row
from distkeras.distributed import serialize_keras_model
from distkeras.distributed import deserialize_keras_model

import os

def rest_post(host, port, endpoint, data):
    request = urllib2.Request("http://" + host + ":" + `port` + endpoint,
                              pickle.dumps(data, -1),
                              headers={'Content-Type': 'application/dist-keras'})

    return urllib2.urlopen(request).read()

def rest_get(host, port, endpoint):
    request = urllib2.Request("http://" + host + ":" + `port` + endpoint,
                              headers={'Content-Type': 'application/dist-keras'})

    return pickle.loads(urllib2.urlopen(request).read())


class EASGD(Trainer):

    def __init__(self, keras_model, features_col="features", label_col="label", num_workers=2):
        super(EASGD, self).__init__(keras_model=keras_model)
        self.features_column = features_col
        self.label_column = label_col
        self.num_workers = num_workers
        # Initialize attribute which do not change throughout the training process.
        self.mutex = Lock()
        # Initialize default parameters.
        self.reset()

    def reset(self):
        # Reset the training attributes.
        self.model = deserialize_keras_model(self.master_model)
        self.gradients = {}
        self.service = None
        self.ready = False
        self.iteration = 0

    def set_ready(self, state):
        with self.mutex:
            self.ready = state

    def get_ready(self):
        localReady = None
        with self.mutex:
            localReady = self.ready

        return localReady

    def start_service(self):
        self.service = Process(target=self.easgd_service)
        self.service.start()

    def stop_service(self):
        self.service.stop()
        self.service.join()

    def process_gradients(self):
        print("\n\n\n--- Processing Gradients in iteration " + `self.iteration` + "---\n\n\n")

    def train(self, data):
        # Start the EASGD REST API.
        self.start_service()
        # Specify the parameters to the worker method.
        worker = EASGDWorker(keras_model=self.master_model,
                             features_col=self.features_column,
                             label_col=self.label_column)
        # Prepare the data, and start the distributed training.
        data.repartition(self.num_workers)
        data.rdd.mapPartitionsWithIndex(worker.train).collect()
        # Stop the EASGD REST API.
        self.stop_service()

        return self.model

    def easgd_service(self):
        app = Flask(__name__)

        ## BEGIN REST routes.###################################################

        @app.route("/center_variable", methods=['GET'])
        def center_variable():
            with self.mutex:
                center_variable = self.model.get_weights()

            return pickle.dumps(center_variable, -1)

        @app.route("/update", methods=['POST'])
        def update():
            data = pickle.loads(request.data)
            gradient = data['gradient']
            worker_id = data['worker_id']

            # Gradient update, declare next iteration.
            self.set_ready(False)
            # Store the gradient of the worker.
            self.gradients[worker_id] = gradient
            # Check if the gradients of all workers are available.
            if len(self.gradients) == self.num_workers:
                self.process_gradients()
                self.gradients = {}
                self.set_ready(True)
                self.iteration += 1

            return 'OK'

        @app.route("/ready", methods=['GET'])
        def ready():
            ready = self.get_ready()

            return pickle.dumps(ready, -1)

        ## END REST routes. ####################################################

        app.run(host='0.0.0.0', threaded=True, use_reloader=False)


class EASGDWorker(object):

    def __init__(self, keras_model, features_col="features", label_col="label", batch_size=1000):
        self.model = deserialize_keras_model(keras_model)
        self.features_column = features_col
        self.label_column = label_col
        self.master_host = "127.0.0.1"
        self.master_port = 5000
        self.master_variable = None
        self.batch_size = batch_size

    def master_send_gradient(self, worker_id, gradient):
        data = {}
        data['worker_id'] = worker_id
        data['gradient'] = gradient
        rest_post(self.master_host, self.master_port, "/update", data)

    def master_is_ready(self):
        return rest_get(self.master_host, self.master_port, "/ready")

    def fetch_center_variable(self):
        self.center_variable = rest_get(self.master_host, self.master_port, "/center_variable")

    def train(self, index, iterator):
        # Compile the model.
        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])
        for i in range(0, 20):
            # Fetch the master (center) variable.
            self.fetch_center_variable()
            # TODO Compute gradient here.
            time.sleep(5)
            # Send the computed gradient to the master.
            self.master_send_gradient(index, gradient)
            # Loop until the master is ready.
            while not self.master_is_ready():
                time.sleep(1)

        return iter([])

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
labelIndexer = StringIndexer(inputCol="Label", outputCol="label_index").fit(dataset)
dataset = labelIndexer.transform(dataset)
# Feature normalization.
standardScaler = StandardScaler(inputCol="features", outputCol="features_normalized", withStd=True, withMean=True)
standardScalerModel = standardScaler.fit(dataset)
dataset = standardScalerModel.transform(dataset)

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

# Sample the dataset.
dataset = dataset.sample(True, 0.01)

# Transform the indexed label to an vector.
labelVectorTransformer = LabelVectorTransformer(output_dim=nb_classes, input_col="label_index", output_col="label")
dataset = labelVectorTransformer.transform(dataset).toDF().select("features_normalized", "label_index", "label")
dataset.printSchema()

# Split the data in a training and test set.
(trainingSet, testSet) = dataset.randomSplit([0.7, 0.3])

# Create the distributed Ensemble trainer.
trainer = EASGD(model, features_col="features_normalized", label_col="label", num_workers=2)
models = trainer.train(trainingSet)
# Get the model from the tuple.
model = models[0][1]
print(model)

# Apply the model prediction to the dataframe.
predictorTransformer = ModelPredictor(keras_model=model, features_col="features_normalized")
testSet = predictorTransformer.predict(testSet).toDF()
testSet.printSchema()
testSet.cache()

# Apply the label index transformer, which will transform the output vector to an indexed label.
indexTransformer = LabelIndexTransformer(output_dim=nb_classes)
testSet = indexTransformer.transform(testSet).toDF()
testSet.printSchema()

# Evaluate the classifier using the MulticlassClassifierEvaluation form Spark's interals
predictionAndLabels = testSet.select("predicted_index", "label_index")
evaluator = MulticlassClassificationEvaluator(metricName="f1", predictionCol="predicted_index", labelCol="label_index")
print("F1: " + str(evaluator.evaluate(predictionAndLabels)))
