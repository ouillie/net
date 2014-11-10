
from libc.stdlib cimport malloc, realloc, free
from libc.math cimport log, sqrt
from cython cimport boundscheck
from cython.parallel cimport prange
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

# Small function for computing modulus of potentially negative numbers using a for-loop (for consistency across implementations).
cdef uint64_t neg_mod(int a, uint64_t n) nogil:
    while a < 0:
        a += n
    while a >= n:
        a -= n
    return a

# Builtin action potential functions.
def sig(double t):
    """Returns the value of the logistic curve evaluated at t."""
    return _sig(t)
def bin(double t):
    """Returns 1 if t > 0 else 0."""
    return _bin(t)

# Built-in cost functions.
@boundscheck(False)
def quad_cost(np.ndarray[double] a, np.ndarray[double] e):
    """Returns the quadratic cost between two vectors."""
    if a.size != e.size:
        raise ValueError('Vectors must have the same dimension.')
    cdef double r = 0.0, d
    cdef uint64_t s = <uint64_t>a.size, i
    for i in range(s):
        d = a[i] - e[i]
        r += d * d
    return r / 2.0
@boundscheck(False)
def cent_cost(np.ndarray[double] a, np.ndarray[double] e):
    """Returns the cross-entropy cost between two vectors."""
    if a.size != e.size:
        raise ValueError('Vectors must have the same dimension.')
    cdef double r = 0.0
    cdef uint64_t s = <uint64_t>a.size, i
    for i in range(s):
        r -= e[i] * log(a[i]) + (1.0 - e[i]) * log(1.0 - a[i])
    return r

# For looking up built-in action potential functions by name.
cdef nodefn cfns(str name):
    if name == 'sig':
        return &_sig
    if name == 'bin':
        return &_bin
    return NULL
# For looking up built-in action potential function names by function pointer.
cdef str re_cfns(nodefn fn):
    if fn == &_sig:
        return 'sig'
    if fn == &_bin:
        return 'bin'
    return None

cdef class Node:
    """The basic element of a Network."""

    cdef double pot[2]                # Potential values at clock low and high.
    cdef object pfn                   # Custom APF.
    cdef double (*cfn)(double) nogil  # Built-in APF.
    cdef bint is_neuron               # Whether to perform backpropagation.
    cdef bint is_input                # Whether to rewind source after training.

    property fn:
        "The action potential function of the node.\n"
        "Built-in APFs are represented by their name as a string. Custom APFs must be callable Python objects.\n"
        "Functions must take a single numerical argument and return a numerical result."
        def __get__(self):
            if self.cfn != NULL:
                return re_cfns(self.cfn)
            return self.pfn
        def __set__(self, object x):
            if x is None:
                self.pfn = None
                self.cfn = NULL
            elif isinstance(x, basestring):
                self.pfn = None
                self.cfn = cfns(str(x))
                if self.cfn == NULL:
                    raise ValueError('APF \'%s\' not a valid built-in function.' % str(x))
            elif callable(x):
                self.pfn = x
                self.cfn = NULL
            else:
                raise TypeError('Expects fn to be callable or a string.')
        def __del__(self):
            self.pfn = None
            self.cfn = NULL

    def __init__(self, double value=0.0, object fn=None):
        """Create a new Node object.

        Keyword arguments:
        value -- A constant potential value.
        fn -- The APF; Node outputs fn(value) if fn is not None else (value).
        """
        self.pot[0] = value
        self.pot[1] = value
        self.fn = fn

    def __call__(self, clock=None):
        """Return the current potential value.

        Keyword arguments:
        clock -- The clock value at which to sample potential. Must be zero or one or None.
                 If None, return the average of the two.
        """
        if clock is None:
            return (self.pot[0] + self.pot[1]) / 2.0
        elif isinstance(clock, int):
            if clock == 0 or clock == 1:
                return self.pot[clock]
            else:
                raise ValueError('clock must be 0 or 1.')
        else:
            raise TypeError('clock must be 0 or 1.')

    # For convenience: maps a Node's input to it's output through it's APF.
    cdef void _update_set(self, bint clock, double pot) nogil:
        if self.pfn is not None:
            with gil:
                pot = <double>self.pfn(pot)
        elif self.cfn != NULL:
            pot = self.cfn(pot)
        self.pot[not clock] = pot

    cdef void _update(self, bint clock) nogil:
        self._update_set(clock, self.pot[clock])

    # The depth of the node from the input layer. By default, Node is considered an input.
    cdef uint64_t depth(self, Path *path, int *err) nogil:
        return 0

    def __str__(self):
        return 'Node'

    def __repr__(self):
        return str(self)

cdef class Input(Node):
    """A Node designed to quickly read data from an input numpy array."""

    cdef double[:] data    # The input vector
    cdef uint64_t i, size  # Current index and max index
    cdef bint loop         # Whether to loop when i >= size

    def __init__(self, double[:] data=None, object fn=None, double value=0.0, bint loop=0):
        """Create a new Input object for feeding numerical data into a network.

        Keyword arguments:
        data -- The data vector; data is read sequentially from the vector at each update. Default is None.
        fn -- The APF; Input outputs fn(data) if fn is not None else (data). Default is None.
        value -- The initial potential of the node before any data is read. Default is zero.
        loop -- Whether to read data cyclically and continuously. By default, potential becomes zero when there is no more data.
        """
        self.is_input = 1
        self.pot[0] = value
        self.pot[1] = value
        self.fn = fn
        self.loop = loop
        if data.size > 0:
            self.size = data.size
            self.data = data

    @boundscheck(False)
    cdef void _update(self, bint clock) nogil:
        if self.data is not None:
            self._update_set(clock, self.data[self.i])
            self.i += 1
            if self.i >= self.size:
                if self.loop:
                    self.i = 0
                else:
                    self.data = None
        else:
            self._update_set(clock, 0.0)

    def __str__(self):
        return 'Input'

    # For rewinding networks in training.
    cdef void rewind(self, int n) nogil:
        if self.loop:
            self.i = neg_mod(<int>self.i - n, self.size)
        elif n > <int>self.i:
            self.i = 0
        else:
            self.i = self.i - n

    @classmethod
    def Layer(self, double[:,:] data=None, object fn=None, double value=0.0, bint loop=0):
        """Returns a list of Input objects, each of which reads data from the corresponding column of a data matrix.

        Keyword arguments:
        data -- The data matrix; data is read sequentially down the rows at each update. Default is None.
        fn -- The APF for each node; Input outputs fn(data) if fn is not None else (data). Default is None.
        value -- The initial potential of each node before any data is read. Default is zero.
        loop -- Whether each node reads data cyclically and continuously. By default, potential becomes zero when there is no more data.
        """
        cols = data.shape[1]
        return [self(data=data[:,i], fn=fn, value=value, loop=loop) for i in range(cols)]

# Represents a parent (input) to a neuron.
cdef struct Parent:
    double weight   # The synaptic weight.
    PyObject *node  # The parent object.

# Linked list for tracing depth and feed-forwardness.
cdef struct Path:
    PyObject *child  # The Node object at this point in the path.
    Path *nxt

cdef class Neuron(Node):
    """A Node designed to process input from other nodes and perform backpropagation."""

    cdef public double bias      # The node bias.
    cdef Parent *_parents        # Array of parents.
    cdef uint64_t c              # Number of parents.
    cdef double *dCdp            # Batch of partial derivatives (for backpropagation).
    cdef uint64_t _depth         # Depth from input layer (for backpropagation).
    cdef uint64_t _output_depth  # Depth from the output layer (for backpropagation).
    cdef double *xs              # Previous number of input values (for backpropagation). Size proportional to depth from output.
    cdef uint64_t _odi           # Current index of the input (output depth) buffer (for backpropagation).

    def __init__(self, double bias=0.0, dict parents=None, object fn='sig', double value=0.0):
        """Create a new Neuron object for processing data in a network.

        Keyword arguments:
        bias -- The node bias; Neuron value is (w * x + b) where w is synaptic weight, x is input and b is bias. Default is zero.
        parents -- Dictionary of the form {parent: weight} where parent is a Node object and weight is the associated synaptic weight. Default is None.
        fn -- The APF; Neuron outputs fn(value) if fn is not None else (value). Default is the logistic curve.
        value -- The initial potential of the node before any data is processed. Default is zero.
        """
        self.is_neuron = 1
        self.pot[0] = value
        self.pot[1] = value
        self.fn = fn
        self.bias = bias
        self.connect(parents)

    def __dealloc__(self):
        cdef uint64_t i
        for i in range(self.c):
            Py_DECREF(<object>self._parents[i].node)
        free(self._parents)

    def __len__(self):
        """Returns the number of parents."""
        return self.c

    def __getitem__(self, object x):
        """Returns self's weight for x. Unconnected nodes are considered to have zero weight."""
        cdef uint64_t i = self.index(<PyObject*>x)
        return self._parents[i].weight if i < self.c else 0.0

    # TODO: See if compound assignment covers this behavior already.
    def __setitem__(self, object x, double y):
        """Adds y to self's weight for x. Unconnected nodes as considered to have zero weight."""
        if not isinstance(x, Node):
            raise TypeError('Parent must be a Node.')
        cdef uint64_t i = self.index(<PyObject*>x)
        if i < self.c:
            self._parents[i].weight += y
        else:
            self.c += 1
            self._parents = <Parent*>realloc(self._parents, self.c * sizeof(Parent))
            if self._parents == NULL:
                self.c = 0
                raise MemoryError('Not enough memory to reallocate self._parents.')
            self._parents[i].node = <PyObject*>x
            Py_INCREF(x)
            self._parents[i].weight = y


    def __delitem__(self, object x):
        """Disconnects self from x."""
        cdef uint64_t i = self.index(<PyObject*>x), j
        if i < self.c:
            Py_DECREF(<object>self._parents[i].node)
            self.c -= 1
            for j in range(i, self.c):
                self._parents[j] = self._parents[j + 1]
            self._parents = <Parent*>realloc(self._parents, self.c * sizeof(Parent))
            if self._parents == NULL:
                self.c = 0
                raise MemoryError('Not enough memory to reallocate parents.')

    def __contains__(self, object x):
        """Returns True is x is connected to self."""
        return self.index(<PyObject*>x) < self.c

    # For convenience: Returns the index of node in parents else self.c.
    cdef uint64_t index(self, PyObject *node):
        cdef uint64_t i
        for i in range(self.c):
            if self._parents[i].node == node:
                return i
        return self.c

    def parents(self):
        """Returns a dictionary of {node: weight} parents connected to self."""
        cdef dict r = {}
        cdef uint64_t i
        for i in range(self.c):
            r[<object>self._parents[i].node] = float(self._parents[i].weight)
        return r

    def connect(self, dict parents=None):
        """Connects a mapping of {node: weight} parents to self."""
        cdef uint64_t l, i
        if parents is not None:
            l = <uint64_t>len(parents)
            if l > 0:
                self.c += l
                self._parents = <Parent*>realloc(self._parents, self.c * sizeof(Parent))
                if self._parents == NULL:
                    self.c = 0
                    raise MemoryError('Not enough memory to reallocate parents.')
                for (key, value) in parents.items():
                    if not isinstance(key, Node):
                        l = self.c - l
                        for i in range(l, self.c):
                            Py_DECREF(<object>self._parents[i].node)
                        self.c = l
                        self._parents = <Parent*>realloc(self._parents, self.c * sizeof(Parent))
                        raise TypeError('All keys of parents must be Nodes.')
                    l -= 1
                    self._parents[l].node = <PyObject*>key
                    Py_INCREF(key)
                    self._parents[l].weight = <double>value

    cdef void _update(self, bint clock) nogil:
        cdef double pot = self.bias, p_pot
        cdef uint64_t i
        for i in range(self.c):
            p_pot = (<Node>self._parents[i].node).pot[clock]
            if self.xs != NULL:
                self.xs[i * self._output_depth + self._odi] = p_pot
                self._odi += 1
                if self._odi >= self._output_depth:
                    self._odi = 0
            pot += p_pot * self._parents[i].weight
        self._update_set(clock, pot)

    # Calculates the depth of the neuron from the input and output layers and checks for feedback loops.
    # Sets err to -1 if a feedback loop is found, -2 if a memory error occurs. Returns zero on error.
    cdef uint64_t depth(self, Path *path, int *err, bint force=0) nogil:
        cdef uint64_t i, tmp, r = 0
        cdef Path *newp = path
        cdef uint64_t output_depth = 1  # Depth from the output.
        while newp != NULL:
            if newp.child == <PyObject*>self:
                err[0] = -1
                return 0
            newp = newp.nxt
            output_depth += 1
        if output_depth > (<Neuron>self)._output_depth:
            (<Neuron>self)._output_depth = output_depth
        if (<Neuron>self)._depth != 0 and not force:
            return (<Neuron>self)._depth
        newp = <Path*>malloc(sizeof(Path))
        if newp == NULL:
            err[0] = -2
            return 0
        newp.child = <PyObject*>self
        newp.nxt = path
        for i in range((<Neuron>self).c):
            tmp = (<Node>(<Neuron>self)._parents[i].node).depth(newp, err)
            if err[0] != 0:
                break
            if tmp > r:
                r = tmp
        free(newp)
        (<Neuron>self)._depth = r + 1
        return (<Neuron>self)._depth

    # Allocates and initializes the buffers used for backpropagation.
    cdef int _init_backprop(self) except -1:
        self.dCdp = <double*>realloc(self.dCdp, (self.c + 1) * sizeof(double))
        if self.dCdp == NULL:
            raise MemoryError('Not enough memory to allocate self.dCdp.')
        if self._output_depth == 0:
            free(self.xs)
            self.xs = NULL
        else:
            self.xs = <double*>realloc(self.xs, self.c * self._output_depth * sizeof(double))
            if self.xs == NULL:
                free(self.dCdp)
                self.dCdp = NULL
                raise MemoryError('Not enough memory to allocate self.xs.')
        self._odi = 0
        cdef uint64_t i, j = self.c + 1
        for i in range(j):
            self.dCdp[i] = 0.0
        j = self.c * self._output_depth
        for i in range(j):
            self.xs[i] = 0.0
        return 0

    # Performs parrallel recursive backpropagation on a batch. Calls the backpropation function of each of it's parents.
    cdef void _backprop(self, double front, bint clock, uint64_t output_depth) nogil:
        front *= self.pot[clock] * (1.0 - self.pot[clock])
        self.dCdp[0] += front  # dCdb
        cdef uint64_t i
        for i in prange(self.c):
            self.dCdp[i + 1] += self.xs[i * self._output_depth + neg_mod(self._odi - output_depth, self._output_depth)] * front
            if (<Node>self._parents[i].node).is_neuron:
                (<Neuron>self._parents[i].node)._backprop(self._parents[i].weight * front, clock, output_depth + 1)

    # Tunes the neuron's parameters by averaging the partial derivatives calculated over a batch.
    cdef void _register_backprop(self, double alpha, double lamb) nogil:
        self.bias -= alpha * self.dCdp[0]
        self.dCdp[0] = 0.0
        cdef uint64_t i
        for i in range(self.c):
            self._parents[i].weight -= alpha * (self.dCdp[i + 1] + lamb * self._parents[i].weight)
            self.dCdp[i + 1] = 0.0

    # Deallocates the buffers used in backpropagation.
    cdef void _dealloc_backprop(self) nogil:
        free(self.dCdp)
        free(self.xs)
        self.dCdp = NULL
        self.xs = NULL

    def __str__(self):
        return 'Neuron(degree=%d, bias=%f)' % (self.c, self.bias)

# Converts a C array to a Python list.
cdef list a_to_l(PyObject **a, uint64_t c):
    cdef uint64_t i
    return [<object>a[i] for i in range(c)]
# Converts a Python list to a C array. Checks for type.
cdef PyObject **l_to_a(list l, uint64_t *c_out, PyObject **prev, uint64_t prevc) except NULL:
    cdef uint64_t i
    for i in range(prevc):
        Py_XDECREF(prev[i])
    c_out[0] = <uint64_t>len(l)
    if c_out[0] == 0:
        return NULL
    prev = <PyObject**>realloc(prev, c_out[0] * sizeof(PyObject*))
    if prev == NULL:
        raise MemoryError('Not enough memory to reallocate array of Python objects.')
    cdef uint64_t j = 0
    for e in l:
        if not isinstance(e, Node):
            for i in range(j):
                Py_XDECREF(prev[i])
            free(prev)
            raise TypeError('All nodes of a network must be Nodes.')
        prev[j] = <PyObject*>e
        Py_INCREF(e)
        j += 1
    return prev
# Decrements the elements of and frees a C array of Python objects.
cdef void free_a(PyObject **a, uint64_t c):
    cdef uint64_t i
    for i in range(c):
        Py_XDECREF(a[i])
    free(a)

cdef class Network:
    """A structure that facilitates processing and learning on a network of nodes."""

    cdef PyObject **_nodes   # C array of member nodes.
    cdef uint64_t c          # Number of nodes in self._nodes.
    cdef bint clock          # Current clock (for updating).
    cdef PyObject **_output  # C array of the output nodes of the network.
    cdef uint64_t oc         # Number of nodes in self._output.

    property nodes:
        "A list of all the nodes in the network."
        def __get__(self):
            return a_to_l(self._nodes, self.c)
        def __set__(self, list x):
            self._nodes = l_to_a(x, &self.c, self._nodes, self.c)
        def __del__(self):
            free_a(self._nodes, self.c)
            self._nodes = NULL
            self.c = 0
    property output:
        "A list of nodes in the network that are designated outputs."
        def __get__(self):
            return a_to_l(self._output, self.oc)
        def __set__(self, list x):
            self._output = l_to_a(x, &self.oc, self._output, self.oc)
        def __del__(self):
            free_a(self._output, self.oc)
            self._output = NULL
            self.oc = 0

    def __init__(self, list nodes=None, list output=None, bint clock=0):
        """Create a new Network object.

        Keyword arguments:
        nodes -- The nodes that describe the network. Default is None.
        output -- The nodes that are designated outputs. Default is None.
        clock -- The starting clock state. Must be zero or one. Default is zero.
        """
        if nodes is not None:
            self.nodes = nodes
        if output is not None:
            self.output = output
        self.clock = clock

    def __dealloc__(self):
        free_a(self._nodes, self.c)
        free_a(self._output, self.oc)

    # Updates a network exactly once. Each Input uses a new sample of data.
    # Returns the vector from the output nodes as a C array. Returns NULL is there are no outputs.
    cdef double *_update_once(self, PyObject **output, uint64_t oc) nogil:
        cdef double *r = NULL
        cdef uint64_t i
        for i in prange(self.c, nogil=True):
            (<Node>self._nodes[i])._update(self.clock)
        self.clock = not self.clock
        if output != NULL:
            r = <double*>malloc(oc * sizeof(double))
            if r != NULL:
                for i in range(oc):
                    r[i] = (<Node>output[i]).pot[self.clock]
        return r

    def update(self, object output=None, uint64_t times=1):
        """Updates the network and returns a matrix of output values as a numpy array.
        The output matrix has dimmensions (times x len(output)). If either dimmension is zero, None is returned.

        Keyword arguments:
        output -- The nodes that are designated outputs. Default is None.
        times -- The number of times to update the network.
        clock -- The starting clock state. Must be zero or one. Default is zero.
        """
        cdef PyObject **_output
        cdef uint64_t oc
        if output is None:
            _output = self._output
            oc = self.oc
        else:
            _output = l_to_a(output, &oc, NULL, 0)
        cdef np.ndarray[double, ndim=2] r
        if times == 0 or oc == 0:
            r = None
        else:
            r = np.ndarray(shape=(times, oc))
        cdef uint64_t i, j
        cdef double *buff
        for i in range(times):
            buff = self._update_once(_output, oc)
            if buff == NULL and _output != NULL:
                raise MemoryError('Not enough memory to allocate output vector.')
            for j in range(oc):
                r[i][j] = buff[j]
            free(buff)
        if _output != self._output:
            free_a(_output, oc)
        return r

    def depth(self):
        """Calculates the maximum depth of a network, defined to be the maximum path length from input to output.
        Raises ValueError if output list is None. Also checks for feed-forwardness and raises ValueError if a feedback loop is found.
        """
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

    cdef uint64_t _bp_depth(self, uint64_t depth, bint verbose=False):
        if depth == 0:
            depth = <uint64_t>self.depth()
        if verbose:
            print('Depth of network = %d' % depth)
        return depth

    cdef int _bp_init(self, bint verbose=False) except -1:
        if verbose:
            print('Setting up backpropagation buffers ...'),
        cdef uint64_t i, j
        for i in range(self.c):
            if (<Node>self._nodes[i]).is_neuron:
                try:
                    (<Neuron>self._nodes[i])._init_backprop()
                except:
                    if verbose:
                        print('Error.')
                    for j in range(i):
                        if (<Node>self._nodes[j]).is_neuron:
                            (<Neuron>self._nodes[j])._dealloc_backprop()
                    raise
        if verbose:
            print('Done.')
        return 0

    cdef void _bp_prerun(self, uint64_t depth, bint verbose=False):
        cdef uint64_t i
        if depth > 0:
            if verbose:
                print('Pre-running ...'),
            for i in range(depth):
                self._update_once(NULL, 0)
            if verbose:
                print('Done.')

    cdef void _bp_rewind(self, uint64_t depth, bint verbose=False):
        if verbose:
            print('Rewinding ...'),
        cdef uint64_t i
        for i in range(self.c):
            if (<Node>self._nodes[i]).is_input:
                (<Input>self._nodes[i]).rewind(depth)
        if verbose:
            print('Done.')

    cdef void _bp_dealloc(self, bint verbose=False):
        if verbose:
            print('Freeing backpropagation buffers ...'),
        cdef uint64_t i
        for i in prange(self.c, nogil=True):
            if (<Node>self._nodes[i]).is_neuron:
                (<Neuron>self._nodes[i])._dealloc_backprop()
        if verbose:
            print('Done.')

    def backprop(self, np.ndarray[double, ndim=2] expect, uint64_t batch=1, double alpha=1.0, double lamb=0.1, uint64_t depth=0, bint verbose=False):
        """Performs some chill backpropagation on this network.

        Keyword arguments:
        expect -- The expected output matrix as a numpy array. Each row is a sample and each column is an output value.
        batch -- The number of partial derivative estimates to average per parameter. Default is 1.
        alpha -- The learning rate of the network. Default is 1.0.
        lamb -- The weight normalization factor used to mitigate overfitting. Larger lambda -> smaller synaptic weights. Default is 0.1.
        depth -- The specific maximum depth of the network. If zero, it is dynamically calculated. Default is zero.
        verbose -- Whether to be verbose. Default is False.
        """
        depth = self._bp_depth(depth, verbose)
        self._bp_init(verbose)
        self._bp_prerun(depth, verbose)
        alpha /= <double>batch
        #lamb *= <double>batch
        cdef uint64_t l = expect.size / (self.oc * batch)
        if verbose:
            print('Running [%d] {' % l)
        cdef double cost, c
        cdef double g, e, max_g, max_e
        cdef uint64_t max_g_i, max_e_i, correct = 0
        cdef uint64_t i, j, k
        for i in range(l):
            cost = 0.0
            for j in range(batch):
                self._update_once(NULL, 0)
                max_g, max_e = 0.0, 0.0
                max_g_i, max_e_i = 0, -1
                #for k in prange(self.oc, nogil=True):
                for k in range(self.oc):
                    g, e = (<Node>self._output[k]).pot[self.clock], expect[i][k]
                    if g > max_g:
                        max_g_i = k
                        max_g = g
                    if e > max_e:
                        max_e_i = k
                        max_e = e
                    c = g - e
                    if (<Node>self._output[k]).is_neuron:
                        with nogil:
                            (<Neuron>self._output[k])._backprop(c, self.clock, 0)
                    cost += c * c
                if max_g_i == max_e_i:
                    correct += 1
            if verbose:
                print('\tBatch [%d] Cost = %f' % (batch, cost / (2.0 * batch)))
            for k in prange(self.c, nogil=True):
                if (<Node>self._nodes[k]).is_neuron:
                    (<Neuron>self._nodes[k])._register_backprop(alpha, lamb)
        if verbose:
            print('} Done')
            print('Accuracy = %f%%' % (<double>correct / <double>(batch * l)))
        self._bp_rewind(depth, verbose)
        self._bp_dealloc(verbose)

    cdef index(self, object node):
        """Returns the index of node in _nodes else self.c."""
        cdef uint64_t i
        for i in range(self.c):
            if self._nodes[i] == <PyObject*>node:
                return i
        return self.c

    def write(self, filename): # TODO: write
        """Encodes, compresses, and saves the network structure to file."""
        f = open(filename, 'wb')
        f.write(pack('!Q', self.c))
        for i in range(self.c):
            if (<Node>self._nodes[i]).is_neuron:
                rents = (<Neuron>self._nodes[i]).parents().items()
                l = len(rents)
                f.write(pack('!BdQ', 2, (<Neuron>self._nodes[i]).bias, l))
                for i in range(l):
                    n, w = rents[i]
                    f.write(pack('!Qd', self.index(n), float(w)))
            elif (<Node>self._nodes[i]).is_input:
                f.write(pack('!B', 1))
            else:
                f.write(pack('!B', 0))
        f.close()

    @classmethod
    def open(self, filename): # TODO: write
        """Reads, decompresses, and decodes a network structure from a file."""
        f = open(filename, 'rb')
        c, = unpack('!Q', f.read(8))
        nodes = []
        rents = []
        for i in range(c):
            t, = unpack('!B', f.read(1))
            if t == 2:    # Neuron
                b, n = unpack('!dQ', f.read(16))
                r = []
                for j in range(n):
                    p, w = unpack('!Qd', f.read(16))
                    r.append((p, w))
                nodes.append(Neuron(bias=b))
                rents.append(r)
            elif t == 1:  # Input
                nodes.append(Input())
                rents.append(None)
            else:         # Node
                nodes.append(Node())
                rents.append(None)
        for i in range(c):
            if rents[i] is not None:
                nodes[i].connect({nodes[ind]: w for ind, w in rents[i]})
        return Network(nodes=nodes)

    def __str__(self):
        cdef uint64_t i
        r = 'Network {\n'
        for i in range(self.c):
            r += '    ' + str(<object>self._nodes[i]) + '\n'
        return r + '}'

    def __repr__(self):
        return str(self)

    @classmethod
    def Layered(self, object layers, double[:,:] data, bint loop=0):
        """Returns a new Network object in a standard layered feed-forward structure.

        Keyword arguments:
        layers -- A list of layer widths. width i is the number of neurons in layer i from the input.
        data -- The input data to the network. An input layer is created using Input.Layer(data=data, loop=loop)
        loop -- Whether the Inputs read data cyclically and continuously. By default, potential becomes zero when there is no more data.
        """
        nodes = Input.Layer(data=data, loop=loop)
        last_w = len(nodes)
        if last_w == 0:
            raise ValueError('Input data must not be empty.')
        for w in layers:
            if w <= 0:
                raise ValueError('Layer width must be positive.')
            nodes.extend([Neuron(bias=normal(), parents={n: normal(0.0, 1.0/sqrt(last_w)) for n in nodes[-last_w:]}, fn='sig') for i in range(w)])
            last_w = w
        return self(nodes=nodes, output=nodes[-last_w:])

