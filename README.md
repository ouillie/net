net
===
This is a Python extension module written in C and Cython for implementing artificial neural networks and performing supervised machine learning.

###Install
The package source comes with a standard distutils setup script. The basic install command is:
```
python setup.py install
```
from the root directory.

#####Dependencies
This depends on the following packages:
* python >= 2.7
* cython >= 0.17
* numpy >= 1.8.2
* glibc >= 2.13

###Usage
The following program creates and performs backpropagation on a neural network to generate outputs equal to `.5` when all it's inputs equal `1`:
```python
from net import Network
import numpy as np

# Backpropagation estimates partial derivatives by averaging the
# estimates of a certain number of test cases, known as the batch size.
batch_size = 20
# We will run backpropagation on a hundred batches.
train_size = batch_size * 100

# Data is typically modeled as a 2-d numpy array, where rows designate
# observations and columns designate elements of a vector. The input
# and output data we create has the same value at each observation.
in_v = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]] * train_size)
out_v = np.array([[0.5, 0.5, 0.5]] * train_size)

# Create a new layered network with 5 inputs,
# 10 hidden neurons and 3 output neurons.
n = Network.Layered([10, 3], in_v)

n.backprop(expect=out_v, batch=batch_size)
```
