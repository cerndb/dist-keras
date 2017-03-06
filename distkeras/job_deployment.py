"""Module which facilitates job deployment on remote Spark clusters.
This allows you to build models and architectures on, for example, remote
notebook servers, and submit the large scale training job on remote
Hadoop / Spark clusters."""

## BEGIN Imports. ##############################################################

from distkeras.utils import deserialize_keras_model
from distkeras.utils import get_os_username
from distkeras.utils import pickle_object
from distkeras.utils import serialize_keras_model
from distkeras.utils import unpickle_object

from flask import Flask
from flask import request

from os.path import expanduser

from threading import Lock

import base64

import json

import os

import subprocess

import threading

import time

import urllib2

## END Imports. ################################################################

class Punchcard(object):

    def __init__(self, secrets_path="secrets.json", port=80):
        self.application = Flask(__name__)
        self.secrets_path = secrets_path
        self.port = port
        self.mutex = threading.Lock()
        self.jobs = {}

    def read_secrets(self):
        with open(self.secrets_path) as f:
            secrets_raw = f.read()
        secrets = json.loads(secrets_raw)

        return secrets

    def valid_secret(self, secret, secrets):
        num_secrets = len(secrets)
        for i in range(0, num_secrets):
            description = secrets[i]
            if description['secret'] == secret:
                return True
        return False

    def secret_in_use(self, secret):
        return secret in self.jobs

    def set_trained_model(self, job, model):
        with self.mutex:
            self.models[job.get_secret()] = model

    def get_submitted_job(self, secret):
        with self.mutex:
            if self.secret_in_use(secret):
                job = self.jobs[secret]
            else:
                job = None

        return job

    def define_routes(self):

        ## BEGIN Route definitions. ############################################

        @self.application.route('/api/submit', methods=['POST'])
        def submit_job():
            # Parse the incoming JSON data.
            data = json.loads(request.data)
            # Fetch the required job arguments.
            secret = data['secret']
            job_name = data['job_name']
            num_executors = data['num_executors']
            num_processes = data['num_processes']
            data_path = data['data_path']
            trainer = unpickle_object(data['trainer'].decode('hex_codec'))
            # Fetch the parameters for the job.
            secrets = self.read_secrets()
            with self.mutex:
                if self.valid_secret(secret, secrets) and not self.secret_in_use(secret):
                    job = PunchcardJob(secret, job_name, data_path, num_executors, num_processes, trainer)
                    self.jobs[secret] = job
                    job.start()
                    return '', 200

            return '', 403

        @self.application.route('/api/state')
        def job_state():
            secret = request.args.get('secret')
            job = self.get_submitted_job(secret)
            # Check if the job exists.
            if job is not None:
                d = {}
                d['job_name'] = job.get_job_name()
                d['running'] = job.running()
                return json.dumps(d), 200

            return '', 404

        @self.application.route('/api/cancel')
        def cancel():
            secret = request.args.get('secret')
            job = self.get_submitted_job(secret)
            if job is not None and job.running():
                with self.mutex:
                    job.cancel()
                    del self.jobs[secret]

            return '', 200

        @self.application.route('/api/destroy')
        def destroy_job():
            secret = request.args.get('secret')
            job = self.get_submitted_job(secret)
            if job is not None and not job.running():
                with self.mutex:
                    model = self.jobs[secret].get_trained_model()
                    history = self.jobs[secret].get_history()
                    model = pickle_object(serialize_keras_model(model)).encode('hex_codec')
                    history = pickle_object(history).encode('hex_codec')
                    d = {}
                    d['model'] = model
                    d['history'] = history
                    del self.jobs[secret]
                return json.dumps(d), 200

            return '', 400

        ## END Route definitions. ##############################################

    def run(self):
        self.define_routes()
        self.application.run('0.0.0.0', self.port)


class PunchcardJob(object):

    def __init__(self, secret, job_name, data_path, num_executors, num_processes, trainer):
        self.secret = secret
        self.job_name = job_name
        self.data_path = data_path
        self.num_executors = num_executors
        self.num_processes = num_processes
        self.trainer = trainer
        self.is_running = True
        self.thread = None
        self.trained_model = None
        self.history = None

    def get_job_name(self):
        return self.job_name

    def get_secret(self):
        return self.secret

    def get_history(self):
        return self.history

    def get_trained_model(self):
        return self.trained_model

    def start(self):
        self.trainer.determine_new_master()
        self.thread = threading.Thread(target=self.run)
        self.thread.setDaemon(True)
        self.thread.start()

    def cancel(self):
        self.thread.exit()

    def running(self):
        return self.is_running

    def join(self):
        self.thread.join()

    def run_job(self):
        os.system("python ~/jobs/" + self.secret + ".py")

    def clean_up(self):
        home = expanduser("~")
        os.remove(home + "/models/" + self.secret)
        os.remove(home + "/histories/" + self.secret)
        os.remove(home + "/trainers/" + self.secret)

    def read_trained_model(self):
        home = expanduser("~")
        with open(home + "/models/" + self.secret, "r") as f:
            self.trained_model = deserialize_keras_model(unpickle_object(f.read()))

    def read_history(self):
        home = expanduser("~")
        with open(home + "/histories/" + self.secret, "r") as f:
            self.history = unpickle_object(f.read())

    def serialize_trainer(self):
        trainer = pickle_object(self.trainer)
        home = expanduser("~")
        with open(home + "/trainers/" + self.secret, "w") as f:
            f.write(trainer)

    def generate_code(self):
        source = """
from distkeras.evaluators import *
from distkeras.predictors import *
from distkeras.trainers import *
from distkeras.trainers import *
from distkeras.transformers import *
from distkeras.utils import *
from keras import *
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark import SQLContext
from os.path import expanduser
secret = '{secret}'
application_name = '{job_name}'
num_executors = {num_executors}
num_processes = {num_processes}
path_data = '{data_path}'
num_workers = num_processes * num_executors
# Allocate a Spark Context, and a Spark SQL context.
conf = SparkConf()
conf.set("spark.app.name", application_name)
conf.set("spark.master", "yarn-client")
conf.set("spark.executor.cores", num_processes)
conf.set("spark.executor.instances", num_executors)
conf.set("spark.executor.memory", "5g")
conf.set("spark.locality.wait", "0")
conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
# Read the dataset from HDFS. For now we assume Parquet files.
dataset = sqlContext.read.parquet(path_data).repartition(num_workers)
# Deserialize the trainer object.
home = expanduser("~")
with open(home + "/trainers/" + secret, "r") as f:
    trainer = unpickle_object(f.read())
# Train the model, and save it afterwards.
trained_model = trainer.train(dataset)
with open(home + "/models/" + secret, "w") as f:
    f.write(pickle_object(serialize_keras_model(trained_model)))
# Save the history of the training process.
histories = trainer.get_history()
with open(home + "/histories/" + secret, "w") as f:
    f.write(pickle_object(histories))
sc.stop()
        """.format(
            secret=self.secret,
            job_name=self.job_name,
            num_executors=self.num_executors,
            num_processes=self.num_processes,
            data_path=self.data_path
        )
        home = expanduser("~")
        with open(home + "/jobs/" + self.secret + ".py", "w") as f:
            f.write(source)

    def run(self):
        self.serialize_trainer()
        self.generate_code()
        self.run_job()
        self.read_trained_model()
        self.read_history()
        self.clean_up()
        self.is_running = False


class Job(object):

    def __init__(self, secret, job_name, data_path, num_executors, num_processes, trainer):
        self.secret = secret
        self.job_name = job_name
        self.num_executors = 20
        self.num_processes = 1
        self.data_path = data_path
        self.trainer = trainer
        self.trained_model = None
        self.history = None
        self.address = None

    def set_num_executors(self, num_executors):
        self.num_executors = num_executors

    def set_num_processes(self, num_processes):
        self.num_processes = num_processes

    def get_trained_model(self):
        return self.trained_model

    def get_history(self):
        return self.history

    def is_finished(self):
        address = self.address + '/api/state?secret=' + self.secret
        request = urllib2.Request(address)
        response = urllib2.urlopen(request)
        data = json.load(response)

        return not data['running']

    def destroy_remote_job(self):
        address = self.address + '/api/destroy?secret=' + self.secret
        request = urllib2.Request(address)
        response = urllib2.urlopen(request)
        data = json.load(response)
        model = unpickle_object(data['model'].decode('hex_codec'))
        self.trained_model = deserialize_keras_model(model)
        self.history = unpickle_object(data['history'].decode('hex_codec'))

    def start(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def wait_completion(self):
        self.thread.join()

    def cancel(self):
        address = self.address + '/api/cancel?secret=' + self.secret
        request = urllib2.Request(address)
        urllib2.urlopen(request)

    def send(self, address):
        data = {}
        data['secret'] = self.secret
        data['job_name'] = self.job_name
        data['num_executors'] = self.num_executors
        data['num_processes'] = self.num_processes
        data['data_path'] = self.data_path
        data['trainer'] = pickle_object(self.trainer).encode('hex_codec')
        request = urllib2.Request(address + "/api/submit")
        request.add_header('Content-Type', 'application/json')
        urllib2.urlopen(request, json.dumps(data))
        self.address = address
        self.start()

    def run(self):
        time.sleep(1)
        while not self.is_finished():
            time.sleep(10)
        self.destroy_remote_job()
