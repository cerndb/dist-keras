"""A class which describes the properties and actions of an Apache Spark
Keras model. This model will set up the required properties and execute
the actions which are needed to distribute the learning process, given
the specified arguments.
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
