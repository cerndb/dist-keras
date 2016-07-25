"""Module which describes the properties and actions of training a Keras model
on Apache Spark.
"""

## BEGIN Imports. ##############################################################

import distkeras
import distkeras.util.networking

## END Imports. ################################################################

class SparkModel:

    def __init__(self, sc, keras_model, master_port=5000):
        self.spark_context = sc
        self.master_model = keras_model
        self.master_address = determine_host_address()
        self.master_port = master_port
