net
===
This is a Python extension module written in C and Cython for implementing artificial neural networks and performing supervised machine learning.

###Install
This package comes with a standard distutils setup script. The basic install command is:
```
python setup.py install
```
from the root directory.
#####Dependencies
net depends on the following packages:
* python >= 2.7
* cython >= 0.17
* numpy >= 1.8.2
* glibc >= 2.13

###Usage
The following program creates and performs backpropagation on a neural network to generate outputs equal to `.5` when all it's inputs equal `1`:
```python
from net import Network
import numpy as np

batch_size = 20
batch_count = 100

in_v = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]] * (batch_size * batch_count))
out_v = np.array([[0.5, 0.5, 0.5]] * (batch_size * batch_count))

n = Network.Layered([10, 3], in_v)
n.backprop(expect=out_v, batch=batch_size)
```
