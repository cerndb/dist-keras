"""Setup-module for DistKeras.

This software enables distributed Deep Learning on Apache Spark.

See:
https://github.com/JoeriHermans/DistKeras/
http://joerihermans.com/
"""

from setuptools import setup
from setuptools import find_packages

setup(name='dist-keras',
      description='Deep learning on Apache Spark with Keras.',
      url='https://github.com/JoeriHermans/DistKeras',
      author='Joeri Hermans',
      author_email='joeri@joerihermans.com',
      license='MIT',
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 3 - Alpha',
          # Project is intended for:
          'Intended Audience :: Developers',
          'Topic :: Software Development :: Analytics',
          # Python versions
          'Programming Language :: Python :: 2.7',
      ],
      packages=['distkeras'],
      package_data={'distkeras': ['distkeras/*.py'],
                                 ['distkeras/util/*.py']},
      # Keywords related to the project.
      keywords=['Keras', 'Deep Learning', 'Machine Learning', 'Theano', 'Tensorflow', 'Distributed', 'Apache Spark'],
      # Package requirements.
      install_requires=['keras'],
)
