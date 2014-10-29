
from cython.parallel cimport prange
from libc.stdlib cimport malloc, realloc, free
from libc.math cimport log, sqrt
from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF, Py_XDECREF
cimport numpy as np

from numpy.random import normal
from struct import pack, unpack

cdef extern from 'stdint.h':
    ctypedef unsigned long long uint64_t

cdef extern from 'cfns.h':
    ctypedef double (*nodefn)(double) nogil
    extern double _sig(double) nogil
    extern double _bin(double) nogil
    extern int _compare_objects(const void*, const void*) nogil

# Builtin action potential functions
def sig(double t):
    return _sig(t)
def bin(double t):
    return _bin(t)

# Builtin cost functions
def quad_cost(np.ndarray[double] a, np.ndarray[double] e):
    cdef double r = 0.0, d
    cdef uint64_t n1 = <uint64_t>a.size, n2 = <uint64_t>e.size, i
    if n2 < n1:
        n1 = n2
    for i in range(n1):
        d = a[i] - e[i]
        r += d * d
    return r / 2.0
def cent_cost(np.ndarray[double] a, np.ndarray[double] e):
    cdef double r = 0.0
    cdef uint64_t n1 = <uint64_t>a.size, n2 = <uint64_t>e.size, i
    if n2 < n1:
        n1 = n2
    for i in range(n1):
        r -= e[i] * log(a[i]) + (1.0 - e[i]) * log(1.0 - a[i])
    return r

cdef nodefn cfns(str name):
    if name == 'sig':
        return &_sig
    if name == 'bin':
        return &_bin
    return NULL
cdef str re_cfns(nodefn fn):
    if fn == &_sig:
        return 'sig'
    if fn == &_bin:
        return 'bin'
    return None

cdef class Node:

    cdef double pot[2]
    cdef object pfn
    cdef double (*cfn)(double) nogil
    cdef bint is_neuron

    property fn:
        "If fn is a string, use the associated built-in C function. If fn is callable, use it as a python function."
        def __get__(self):
            if self.pfn is not None:
                return self.pfn
            if self.cfn != NULL:
                return re_cfns(self.cfn)
            return None
        def __set__(self, object x):
            if x is None:
                self.pfn = None
                self.cfn = NULL
            elif callable(x):
                self.pfn = x
            elif isinstance(x, basestring):
                self.pfn = None
                self.cfn = cfns(str(x))
            else:
                raise TypeError('Expects fn to be callable or a string.')
        def __del__(self):
            self.pfn = None
            self.cfn = NULL

    def __init__(self, double value=0.0, object fn=None):
        self.pot[0] = value
        self.pot[1] = value
        self.fn = fn

    def __call__(self, bint clock=0):
        return self.pot[clock]

    cdef void _update(self, bint clock) nogil:
        pass

    cdef uint64_t depth(self, Path *path, int *err) nogil:
        return 0

    def __str__(self):
        return 'Node'
    def __repr__(self):
        return str(self)

cdef class Input(Node):

    cdef double[:] data
    cdef uint64_t i, size
    cdef bint loop

    def __init__(self, double[:] data=None, object fn=None, double value=0.0, bint loop=0):
        self.pot[0] = value
        self.pot[1] = value
        self.loop = loop
        self.size = data.size
        self.data = data if self.size > 0 else None
        self.fn = fn

    cdef void _update(self, bint clock) nogil:
        cdef double pot
        if self.data is not None:
            pot = self.data[self.i]
            self.i += 1
            if self.loop:
                self.i %= self.size
            elif self.i >= self.size:
                self.data = None
        else:
            pot = 0.0
        if self.pfn is not None:
            with gil:
                pot = <double>self.pfn(pot)
        elif self.cfn != NULL:
            pot = self.cfn(pot)
        self.pot[not clock] = pot

    def __str__(self):
        return 'Input'

    @classmethod
    def Layer(self, double[:,:] data=None, object fn=None, double value=0.0, bint loop=0):
        cols = data.shape[1]
        return [Input(data=data[i], fn=fn, value=value, loop=loop) for i in range(cols)]

cdef struct Parent:
    PyObject *node
    double bias

cdef struct Path:
    PyObject *child
    Path *nxt

cdef class Neuron(Node):

    cdef public double bias  # the node bias
    cdef Parent *parents     # array of parents
    cdef uint64_t c          # number of parents
    cdef double *dCdp        # for backpropagation

    def __cinit__(self, double bias=0.0, dict parents=None, object fn='sig', double value=0.0):
        self.is_neuron = 1

    def __init__(self, double bias=0.0, dict parents=None, object fn='sig', double value=0.0):
        self.pot[0] = value
        self.pot[1] = value
        self.fn = fn
        self.bias = bias
        self.connect(parents)

    def __dealloc__(self):
        cdef uint64_t i
        for i in range(self.c):
            Py_DECREF(<object>self.parents[i].node)
        free(self.parents)

    def __len__(self):
        """ Returns the number of parents. """
        return self.c

    def __getitem__(self, object x):
        """ Returns self's bias for x. """
        cdef uint64_t i = self.index(<PyObject*>x)
        return self.parents[i].bias if i < self.c else 0.0

    def __setitem__(self, object x, double y):
        """ Adds y to self's bias for x. """
        if not isinstance(x, Node):
            raise TypeError('Parent must be a Node.')
        cdef uint64_t i = self.index(<PyObject*>x)
        if i < self.c:
            self.parents[i].bias += y
        else:
            self.c += 1
            self.parents = <Parent*>realloc(self.parents, self.c * sizeof(Parent))
            if self.parents == NULL:
                self.c = 0
                raise MemoryError('Not enough memory to reallocate self.parents.')
            self.parents[i].node = <PyObject*>x
            Py_INCREF(x)
            self.parents[i].bias = y


    def __delitem__(self, object x):
        """ Disconnects self from x. """
        cdef uint64_t i = self.index(<PyObject*>x), j
        if i < self.c:
            Py_DECREF(<object>self.parents[i].node)
            self.c -= 1
            for j in range(i, self.c):
                self.parents[j] = self.parents[j + 1]
            self.parents = <Parent*>realloc(self.parents, self.c * sizeof(Parent))

    def __contains__(self, object x):
        """ Returns True is x is a parent of self. """
        return self.index(<PyObject*>x) < self.c

    cdef uint64_t index(self, PyObject *node):
        """ Returns the index of node in parents else self.c. """
        cdef uint64_t i
        for i in range(self.c):
            if self.parents[i].node == node:
                return i
        return self.c

    def connect(self, dict parents=None):
        """ Connects a mapping of {node: bias} parents to self. """
        cdef uint64_t l, i
        if parents is not None:
            l = <uint64_t>len(parents)
            if l > 0:
                self.c += l
                self.parents = <Parent*>realloc(self.parents, self.c * sizeof(Parent))
                if self.parents == NULL:
                    self.c = 0
                    raise MemoryError('Not enough memory to reallocate parents.')
                for (key, value) in parents.items():
                    if not isinstance(key, Node):
                        l = self.c - l
                        for i in range(l, self.c):
                            Py_DECREF(<object>self.parents[i].node)
                        self.c = l
                        self.parents = <Parent*>realloc(self.parents, self.c * sizeof(Parent))
                        raise TypeError('All keys of parents must be Nodes.')
                    l -= 1
                    self.parents[l].node = <PyObject*>key
                    Py_INCREF(key)
                    self.parents[l].bias = <double>value

    cdef void _update(self, bint clock) nogil:
        cdef double pot = self.bias
        cdef uint64_t i
        for i in range(self.c):
            pot += (<Node>self.parents[i].node).pot[clock] * self.parents[i].bias
        if self.pfn is not None:
            with gil:
                pot = <double>self.pfn(pot)
        elif self.cfn != NULL:
            pot = self.cfn(pot)
        self.pot[not clock] = pot

    cdef uint64_t depth(self, Path *path, int *err) nogil:
        cdef uint64_t i, tmp, r = 0
        cdef Path *newp = path
        while newp != NULL:
            if newp.child == <PyObject*>self:
                err[0] = -1
                return 0
            newp = newp.nxt
        newp = <Path*>malloc(sizeof(Path))
        if newp == NULL:
            err[0] = -2
            return 0
        newp.child = <PyObject*>self
        newp.nxt = path
        for i in range(self.c):
            tmp = (<Node>self.parents[i].node).depth(newp, err)
            if err[0] != 0:
                break
            if tmp > r:
                r = tmp
        free(newp)
        return r + 1

    cdef int _init_backprop(self) except -1:
        self.dCdp = <double*>malloc((self.c + 1) * sizeof(double))
        if self.dCdp == NULL:
            raise MemoryError('Not enough memory to allocate self.dCdp.')
            return -1
        cdef uint64_t i, j = self.c + 1
        for i in range(j):
            self.dCdp[i] = 0.0
        return 0

    cdef void _backprop(self, double front, bint clock) nogil:
        front *= self.pot[clock] * (1.0 - self.pot[clock])
        self.dCdp[0] += front  # dCdb
        cdef uint64_t i
        for i in prange(self.c):
            self.dCdp[i + 1] += (<Node>self.parents[i].node).pot[clock] * front
            if (<Node>self.parents[i].node).is_neuron:
                (<Neuron>self.parents[i].node)._backprop(self.parents[i].bias * front, clock)

    cdef void _register_backprop(self, double alpha, double lamb) nogil:
        self.bias -= alpha * self.dCdp[0]
        self.dCdp[0] = 0.0
        cdef uint64_t i
        for i in range(self.c):
            self.parents[i].bias -= alpha * (self.dCdp[i + 1] + lamb * self.parents[i].bias)
            self.dCdp[i + 1] = 0.0

    cdef void _dealloc_backprop(self) nogil:
        free(self.dCdp)

    def __str__(self):
        return 'Neuron(degree=%d, bias=%f)' % (self.c, self.bias)

cdef list a_to_l(PyObject **a, uint64_t c):
    cdef uint64_t i
    return [<object>a[i] for i in range(c)]
cdef PyObject **l_to_a(list l, uint64_t *c_out, PyObject **prev, uint64_t prevc) except NULL:
    c_out[0] = <uint64_t>len(l)
    cdef uint64_t i
    for i in range(prevc):
        Py_XDECREF(prev[i])
    prev = <PyObject**>realloc(prev, c_out[0] * sizeof(PyObject*))
    if prev == NULL:
        raise MemoryError('Not enough memory to reallocate array of Python objects.')
        return NULL
    cdef uint64_t j = 0
    for e in l:
        if not isinstance(e, Node):
            for i in range(j):
                Py_XDECREF(prev[i])
            free(prev)
            raise TypeError('All nodes of a network must be Nodes.')
            return NULL
        prev[j] = <PyObject*>e
        Py_INCREF(e)
        j += 1
    return prev
cdef void free_a(PyObject **a, uint64_t c):
    cdef uint64_t i
    for i in range(c):
        Py_XDECREF(a[i])
    free(a)

cdef class Network:

    cdef PyObject **_nodes
    cdef uint64_t c
    cdef bint clock
    cdef PyObject **_output
    cdef uint64_t oc
    cdef uint64_t layers

    property nodes:
        def __get__(self):
            return a_to_l(self._nodes, self.c)
        def __set__(self, list x):
            self._nodes = l_to_a(x, &self.c, self._nodes, self.c)
        def __del__(self):
            free_a(self._nodes, self.c)
            self._nodes = NULL
            self.c = 0
    property output:
        def __get__(self):
            return a_to_l(self._output, self.oc)
        def __set__(self, list x):
            self._output = l_to_a(x, &self.oc, self._output, self.oc)
        def __del__(self):
            free_a(self._output, self.oc)
            self._output = NULL
            self.oc = 0

    def __cinit__(self, list nodes=None, list output=None, bint clock=0):
        self.clock = clock

    def __init__(self, list nodes=None, list output=None, bint clock=0):
        if nodes is not None:
            self.nodes = nodes
        if output is not None:
            self.output = output

    def __dealloc__(self):
        free_a(self._nodes, self.c)
        free_a(self._output, self.oc)

    cdef double *_update_once(self, PyObject **output, uint64_t oc) except? NULL:
        cdef double *r = NULL
        cdef uint64_t i
        for i in prange(self.c, nogil=True):
            (<Node>self._nodes[i])._update(self.clock)
        self.clock = not self.clock
        if output != NULL:
            r = <double*>malloc(oc * sizeof(double))
            if r == NULL:
                raise MemoryError('Not enough memory to reallocate parents.')
            else:
                for i in range(oc):
                    r[i] = (<Node>output[i]).pot[self.clock]
        return r

    def update(self, object output=None, uint64_t times=1):
        cdef PyObject **_output
        cdef uint64_t oc
        if output is None:
            _output = self._output
            oc = self.oc
        else:
            _output = l_to_a(output, &oc, NULL, 0)
        cdef np.ndarray[double, ndim=2] r = np.ndarray(shape=(times, oc))
        cdef uint64_t i, j
        cdef double *buff
        for i in range(times):
            buff = self._update_once(_output, oc)
            for j in range(oc):
                r[i][j] = buff[j]
            free(buff)
        if _output != self._output:
            free_a(_output, oc)
        return r

    def depth(self):
        if self._output == NULL:
            raise ValueError('Network is not feed-forward.')
        cdef uint64_t depth = 0, i, tmp
        cdef int err = 0
        for i in range(self.oc):
            tmp = (<Node>self._output[i]).depth(NULL, &err)
            if err == -1:
                raise ValueError('Network is not feed-forward.')
            if err == -2:
                raise MemoryError('Not enough memory to allocate search path.')
            if tmp > depth:
                depth = tmp
        return depth

    def backprop(self, np.ndarray[double, ndim=2] expect, uint64_t batch=1, double alpha=0.5, double lamb=0.1, uint64_t depth=0):
        if depth == 0:
            print('Calculating depth of network ='),
            depth = <uint64_t>self.depth()
            print('%d' % depth)
        cdef uint64_t i, j, k
        if depth > 0:
            print('Pre-running ...'),
            for i in range(depth):
                self._update_once(NULL, 0)
            print('Done')
        print('Setting up backpropagation buffers ...'),
        for i in range(self.c):
            if (<Node>self._nodes[i]).is_neuron:
                try:
                    (<Neuron>self._nodes[i])._init_backprop()
                except:
                    for j in range(i):
                        if (<Node>self._nodes[j]).is_neuron:
                            (<Neuron>self._nodes[j])._dealloc_backprop()
                    raise
        print('Done')
        alpha /= <double>batch
        cdef uint64_t l = expect.size / (self.oc * batch)
        print('Running [%d] {' % l)
        cdef double cost, c
        for i in range(l):
            cost = 0.0
            for j in range(batch):
                self._update_once(NULL, 0)
                for k in range(self.oc):
                    c = (<Node>self._output[k]).pot[self.clock] - expect[i][k]
                    if (<Node>self._output[k]).is_neuron:
                        with nogil:
                            (<Neuron>self._output[k])._backprop(c, self.clock)
                    cost += c * c
            print('\tBatch [%d] Cost = %f' % (batch, cost / (2.0 * batch)))
            for k in prange(self.c, nogil=True):
                if (<Node>self._nodes[k]).is_neuron:
                    (<Neuron>self._nodes[k])._register_backprop(alpha, lamb)
        print('} Done')
        print('Freeing backpropagation buffers ...'),
        for i in prange(self.c, nogil=True):
            if (<Node>self._nodes[i]).is_neuron:
                (<Neuron>self._nodes[i])._dealloc_backprop()
        print('Done')

    def write(self, filename):
        f = open(filename, 'wb')
        f.write(pack('', self.c)) # TODO: write
        f.close()

    @classmethod
    def open(self, filename): # TODO: write
        nodes = []
        output = []
        cdef bint clock = 0
        return Network(nodes=nodes, output=output, clock=clock)

    def __str__(self):
        cdef uint64_t i
        r = 'Network {\n'
        for i in range(self.c):
            r += '    ' + str(<object>self._nodes[i]) + '\n'
        return r + '}'

    def __repr__(self):
        return str(self)

    @classmethod
    def Layered(self, object layers, double[:,:] data):
        nodes = Input.Layer(data=data)
        last_w = len(nodes)
        if last_w == 0:
            raise ValueError('Input data must not be empty.')
        for w in layers:
            if w <= 0:
                raise ValueError('Layer width must be positive.')
            nodes.extend([Neuron(bias=normal(), parents={n: normal(0.0, 1.0/sqrt(last_w)) for n in nodes[-last_w:]}, fn='sig') for i in range(w)])
            last_w = w
        return self(nodes=nodes, output=nodes[-last_w:])

