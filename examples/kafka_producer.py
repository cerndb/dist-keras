"""This example will be used as a Kafka producer to generate dummy
data for our Spark Streaming example.
"""

## BEGIN Imports. ##############################################################

from kafka import *

import sys

import pandas

import time

import json

## END Imports. ################################################################

def usage():
    print("Distributed Keras Example: Kafka Producer")
    print("")
    print("Usage:")
    print("python kafka_producer.py [bootstrap_server]")
    exit()

def allocate_producer(bootstrap_server):
    producer = KafkaProducer(bootstrap_servers=[bootstrap_server])

    return producer

def read_data():
    path = 'data/atlas_higgs.csv'
    data = []
    # Use Pandas to infer the types.
    data = pandas.read_csv(path)
    # Remove the unneeded columns.
    del data['Label']
    del data['Weight']
    # Convert the data to a list of dictionaries.
    data = data.transpose().to_dict().values()

    return data

def produce(producer, topic, data):
    for row in data:
        producer.send(topic, json.dumps(row))

def main():
    # Check if the required number of arguments has been specified.
    if len(sys.argv) != 2:
        usage()
    # Fetch the bootstrap server from the arguments.
    bootstrap_server = sys.argv[1]
    # Allocate the producer.
    producer = allocate_producer(bootstrap_server)
    # Read the data from the CSV file.
    data = read_data()
    iteration = 1
    # Transmit the data in a continous loop while waiting for 5 seconds after every iteration.
    while True:
        print("Iteration " + str(iteration) + ".")
        produce(producer, 'Machine_Learning', data)
        iteration += 1
        time.sleep(5)

if __name__ == "__main__":
    main()
