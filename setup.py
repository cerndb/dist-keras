"""Setup-module for DistKeras.

This software enables distrubuted Machine Learning on Apache Spark using Keras.

See:
https://github.com/JoeriHermans/dist-keras/
http://joerihermans.com/
"""

from setuptools import setup
from setuptools import find_packages

setup(name='dist-keras',
      description='Distributed Deep learning with Apache Spark with Keras.',
      url='https://github.com/JoeriHermans/dist-keras',
      author='Joeri Hermans',
      version='0.2.0',
      author_email='joeri@joerihermans.com',
      license='GPLv3',
      install_requires=['theano', 'tensorflow', 'keras', 'flask'],
      packages=['distkeras'],
      package_data={'distkeras': ['distkeras/*.py']},
      # Keywords related to the project.
      keywords=['Keras', 'Deep Learning', 'Machine Learning', 'Theano', 'Tensorflow', 'Distributed', 'Apache Spark'],
)
