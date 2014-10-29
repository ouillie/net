#!/usr/bin/python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

net_ext = Extension('net', ['net.pyx', 'cfns.c'])

setup(name='net',
      version='0.1',
      description='Fast Neural Net Simulator',
      author='Will Noble',
      author_email='wnoble@mit.edu',
      include_dirs=[np.get_include()],
      ext_modules=cythonize(net_ext)
)

