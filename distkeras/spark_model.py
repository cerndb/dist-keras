"""Module which describes the properties and actions of training a Keras model
on Apache Spark.
"""

## BEGIN Imports. ##############################################################

import socket

## END Imports. ################################################################

class SparkModel:

    def __init__(self, sc, keras_model, master_port=5000):
        self.spark_context = sc
        self.master_model = keras_model
        self.master_address = self.determine_master_address()
        self.master_port = master_port

    @staticmethod
    def determine_master_address():
        master_address = socket.gethostbyname(socket.gethostname())

        return master_address
