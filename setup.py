
import pyplasma
import os
import sys
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open("README.md", 'r') as file:
    long_description = file.read()

setup(name = pyplasma.__name__,
      version = pyplasma.__version__,
      author = pyplasma.__author__,
      author_email = 'jean-luc.deziel.1@ulaval.ca',
      url = 'https://github.com/jldez/pyplasma',
      description = 'Python module for plasma formation modelling.',
      long_description = long_description,
      long_description_content_type = 'text/markdown',
      packages = ['pyplasma'],
      install_requires = ['numpy','matplotlib','scipy','tqdm'],
    )