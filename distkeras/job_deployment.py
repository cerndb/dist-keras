"""Module which facilitates job deployment on remote Spark clusters.
This allows you to build models and architectures on, for example, remote
notebook servers, and submit the large scale training job on remote
Hadoop / Spark clusters."""

## BEGIN Imports. ##############################################################

from distkeras.utils import get_os_username
from distkeras.utils import serialize_keras_model

from flask import Flask

from threading import Lock

import json

import os

import subprocess

import threading

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
            secrets = json.loads(f.read())

        return secrets

    def valid_secret(self, secret, secrets):
        for k in secrets:
            description = secrets[k]
            if description['secret'] == secret:
                return True
        return False

    def secret_in_use(self, secret):
        return secret in self.jobs

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
        def submit_job(self):
            # Parse the incoming JSON data.
            data = json.loads(request.data)
            # Fetch the required job arguments.
            secret = data['secret']
            job_name = data['job_name']
            num_executors = data['num_executors']
            num_processes = data['num_processes']
            data_path = data['data_path']
            trainer = unpickle_object(data['trainer'])
            # Fetch the parameters for the job.
            secrets = self.read_secrets()
            with self.mutex:
                if self.valid_selfecret(secret, secrets) and not self.secret_in_use(secret):
                    job = Job(secret, job_name, data_path, num_executors, num_processes, trainer)
                    self.jobs[secret] = job
                    job.start()

        @self.application.route('/api/state')
        def job_state(self):
            secret = request.args.get('secret')
            job = self.get_submitted_job(secret)
            # Check if the job exists.
            if job is not None:
                print(job.is_running())
                raise NotImplementedError

        @self.application.route('/api/destroy')
        def destroy_job(self):
            secret = request.args.get('secret')
            job = self.get_submitted_job(secret)
            if job is not None and not job.is_running():
                with self.mutex:
                    del self.jobs[job]

        ## END Route definitions. ##############################################

    def run(self):
        self.define_routes()
        self.application.run('0.0.0.0', self.port)


class Job(object):

    def __init__(self, secret, job_name, data_path, num_executors, num_processes, trainer):
        self.secret = secret
        self.job_name = job_name
        self.num_executors = 20
        self.num_processes = 1
        self.data_path = data_path
        self.trainer = trainer
        self.is_running = True
        self.thread = None

    def get_secret(self):
        return self.secret

    def is_running(self):
        return self.is_running

    def get_data_path(self):
        return self.data_path

    def set_num_executors(self, num_executors):
        self.num_executors = num_executors

    def set_num_processes(self, num_processes):
        self.num_processes = num_processes

    def num_executors(self):
        return self.num_executors

    def num_processes(self):
        return self.num_processes

    def get_trainer(self):
        return self.trainer

    def start(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def join(self):
        self.thread.join()

    def send(self, address):
        # Prepare the data to send.
        data = {}
        data['secret'] = self.secret
        data['job_name'] = self.job_name
        data['num_executors'] = self.num_executors
        data['num_processes'] = self.num_processes
        data['data_path'] = self.data_path
        data['trainer'] = pickle_object(self.trainer)
        # Prepare the request.
        request = urllib2.Request(address + "/api/submit")
        request.add_header('Content-Type', 'application/json')
        # Submit the request.
        response = urllib2.urlopen(request, json.dumps(d))

    def generate_code(self):
        raise NotImplementedError

    def execute_job(self):
        raise NotImplementedError

    def process_result(self):
        raise NotImplementedError

    def run(self):
        self.generate_code()
        self.execute_job()
        self.process_result()
        # Job done, set flag.
        self.is_running = False
