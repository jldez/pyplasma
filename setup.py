#!/usr/bin/env python

import os
import sys
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name="pyplasma",

      version = '0.2',
      description='Python module for plasma formation modelling.',
      author='Jean-Luc Déziel',
      author_email='jean-luc.deziel.1@ulaval.ca',
      packages=['pyplasma'],

      install_requires = ['numpy','matplotlib', 'scipy', 'tqdm'],
    )