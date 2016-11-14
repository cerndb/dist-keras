"""This example will be used as a Kafka producer to generate dummy
data for our Spark Streaming example.
"""

## BEGIN Imports. ##############################################################

from kafka import *

import sys

import csv

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
    # Start processing the CSV file.
    with open(path, 'r') as file:
        reader = csv.DictReader(file, delimiter=',')
        # Iterate through the rows.
        for row in reader:
            # Remove features.
            del row['EventId']
            del row['Weight']
            del row['Label']
            # Append processed data.
            data.append(row)

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
