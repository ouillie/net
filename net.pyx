# net.pyx: This monolithic Cython file implements the vast majority of functionality :/

from libc.stdlib cimport malloc, realloc, free
from libc.math cimport log, sqrt
from cython cimport boundscheck
from cython.parallel cimport prange
from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF, Py_XDECREF
cimport numpy as np 
#from pthread cimport *
from numpy.random import normal
from struct import pack, unpack

cdef extern from 'stdint.h':
    ctypedef unsigned long long uint64_t
    ctypedef signed long long int64_t

cdef extern from 'pthread.h':
    ctypedef long pthread_t
    extern struct pthread_attr_t:
        pass
    ctypedef long pthread_rwlock_t
    extern struct pthread_rwlockattr_t:
        pass
    extern int pthread_create(pthread_t*, const pthread_attr_t*, void *(*)(void*), void*) nogil
    extern int pthread_join(pthread_t, void**) nogil
    extern int pthread_rwlock_rdlock(pthread_rwlock_t*) nogil
    extern int pthread_rwlock_wrlock(pthread_rwlock_t*) nogil
    extern int pthread_rwlock_unlock(pthread_rwlock_t*) nogil
    extern int pthread_rwlock_init(pthread_rwlock_t*, pthread_rwlockattr_t*) nogil

cdef extern from 'cfns.h':
    ctypedef double (*nodefn)(double) nogil
    extern double _sig(double) nogil
    extern double _bin(double) nogil

# Small function for computing modulus of potentially negative numbers using a for-loop (for consistency across implementations).
cdef uint64_t neg_mod(int64_t a, uint64_t n) nogil:
    while a < 0:
        a += <int64_t>n
    while a >= <int64_t>n:
        a -= <int64_t>n
    return <uint64_t>a

# Built-in action potential functions.
def sig(double t):
    """Returns the value of the logistic curve evaluated at t."""
    return float(_sig(t))
def bin(double t):
    """Returns 1 if t > 0 else 0."""
    return float(_bin(t))

# Built-in cost functions.
#cdef uint64_t vector_check(double[:] a, double[:] e) except? 0:
#    cdef uint64_t r = <uint64_t>a.size
#    if r != a.size or
@boundscheck(False)
def quad_cost(double[:] a, double[:] e):
    """Returns the quadratic cost between two vectors."""
    cdef uint64_t s = <uint64_t>a.shape[0], i
    if s != <uint64_t>e.shape[0]:
        raise ValueError('Vectors must have the same dimension.')
    cdef double r = 0.0, d
    for i in range(s):
        d = a[i] - e[i]
        r += d * d
    r /= 2.0
    return float(r)
@boundscheck(False)
def cent_cost(double[:] a, double[:] e):
    """Returns the cross-entropy cost between two vectors."""
    cdef uint64_t s = <uint64_t>a.shape[0], i
    if s != <uint64_t>e.shape[0]:
        raise ValueError('Vectors must have the same dimension.')
    cdef double r = 0.0
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

# Linked list structure for tracing depth and checking feed-forwardness.
cdef struct Path:
    PyObject *child  # The Node object at this point in the path from the output.
    Path *nxt

# Common function to Node and Neuron output depth calculation / feedback loop detection.
# Returns -1 if a feedback loop is found, -2 if the lock can't be attained, zero otherwise.
cdef int _calc_output_depth(PyObject *node, Path *path) nogil:
    cdef Path *tmp = path
    cdef uint64_t output_depth = 1
    while tmp != NULL:
        if tmp.child == node:  # Feedback loop found.
            return -1
        tmp = tmp.nxt
        output_depth += 1
    return (<Node>node)._enable_mem(output_depth)

cdef class Node:
    """The basic element of a Network."""

    cdef double pot[2]                # Potential values at clock low and high.
    cdef object pfn                   # Custom APF.
    cdef double (*cfn)(double) nogil  # Built-in APF.
    cdef bint is_neuron               # Whether to perform backpropagation (is subclass of Neuron).
    cdef bint is_input                # Whether to cap input during training (is subclass of Input).
    cdef double *pot_list             # Record of past output values.
    cdef uint64_t pl_n                # Number of past output values.
    cdef uint64_t pl_i                # Current index of past output values.
    cdef uint64_t _output_depth       # Depth from the output layer (for training).
    cdef pthread_rwlock_t lock        # For read/write locking on parallel applications.

    property fn:
        "The action potential function of the node.\n"
        "Built-in APFs are represented by their name as a string. Custom APFs must be callable Python objects.\n"
        "Functions must take a single numerical argument and return a numerical result."
        def __get__(self):
            cdef object r
            if pthread_rwlock_rdlock(&self.lock) != 0:
                raise RuntimeError('Couldn\'t get read lock.')
            try:
                if self.cfn != NULL:
                    r = re_cfns(self.cfn)
                else:
                    r = self.pfn
            finally:
                pthread_rwlock_unlock(&self.lock)
            return r
        def __set__(self, object x):
            if pthread_rwlock_wrlock(&self.lock) != 0:
                raise RuntimeError('Couldn\'t get write lock.')
            try:
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
            finally:
                    pthread_rwlock_unlock(&self.lock)
        def __del__(self):
            if pthread_rwlock_wrlock(&self.lock):
                raise RuntimeError('Couldn\'t get write lock.')
            self.pfn = None
            self.cfn = NULL
            pthread_rwlock_unlock(&self.lock)

    def __init__(self, double value=0.0, object fn=None):
        """Create a new Node object.

        Keyword arguments:
        value : float - A constant potential value. Default is 0.0.
        fn: str or callable - The APF; Node outputs fn(value) if fn is not None else (value).
            if fn is a string, the associated built-in function is used. Default is None.
        """
        if pthread_rwlock_init(&self.lock, NULL) != 0:
            raise RuntimeError('Can\'t initialize read/write lock.')
        self.pot[0] = value
        self.pot[1] = value
        self.fn = fn

    def __call__(self, int past=0):
        """Return the potential value at a certain time in the past (or present).

        Keyword arguments:
        past : int - How many cycles ago to sample the potential. Must be within the limit for this node.
            Zero is the current potential. Default is zero.
        """
        cdef double r
        if pthread_rwlock_rdlock(&self.lock):
            raise RuntimeError('Couldn\'t get read lock.')
        try:
            if past < 0 or past >= self.pl_n:
                raise ValueError('past < 0 or past >= self.pl_n.')
            r = self.pot_list[past]
        finally:
            pthread_rwlock_unlock(&self.lock)
        return r

    # Returns zero; what is considered to be the input depth by default for Nodes and Inputs (Inputs by definition).
    # Also initializes the buffer for recording past output values necesarry in training.
    # Sets err to -1 if a feedback loop is found, -2 if the lock can't be attained, -3 if a memory error occurs. Returns zero on error.
    # NOTE: does not raise exceptions!
    cdef uint64_t depth(self, Path *path, int *err, bint force=False) nogil:
        err[0] = _calc_output_depth(<PyObject*>self, path)
        return 0

    # Returns -2 if the lock can't be attained, -3 if a memory error occurs. Zero on success.
    cdef int _enable_mem(self, uint64_t n) nogil:
        cdef uint64_t i
        if pthread_rwlock_wrlock(&self.lock) != 0:
            return -2
        if n > self.pl_n:
            self.pot_list = <double*>realloc(self.pot_list, n * sizeof(double))
            if self.pot_list == NULL:
                self.pl_n = 0
                return -3
            for i in range(self.pl_n, n):
                self.pot_list[i] = 0.0
            self.pl_n = n
        pthread_rwlock_unlock(&self.lock)
        return 0
    def enable_mem(self, int n):
        if n < 0:
            raise ValueError('n must be non-negative.')
        self._enable_mem(<uint64_t>n)

    cdef int _clear_mem(self) nogil:
        if pthread_rwlock_wrlock(&self.lock) != 0:
            return -2
        free(self.pot_list)
        self.pot_list = NULL
        self.pl_n = 0
        pthread_rwlock_unlock(&self.lock)
        return 0
    def clear_mem(self):
        if self._clear_mem() == -2:
            raise RuntimeError('Couldn\'t get the write lock.')

    # Maps a Node's input to it's output through it's APF.
    # Expects the caller to have a write lock!
    cdef void _update_set(self, bint clock, double pot) nogil:
        if self.cfn != NULL:
            pot = self.cfn(pot)
        elif self.pfn is not None:
            with gil:
                pot = <double>self.pfn(pot)
        self.pot[not clock] = pot
        if self.pot_list != NULL:
            self.pot_list[self.pl_i] = pot
            self.pl_i += 1
            if self.pl_i >= self.pl_n:
                self.pl_i = 0

    # NOTE: does not throw exceptions.
    cdef void _update(self, bint clock) nogil:
        if pthread_rwlock_wrlock(&self.lock) == 0:
            self._update_set(clock, self.pot[clock])
            pthread_rwlock_unlock(&self.lock)

    def __str__(self):
        return 'Node'

    def __repr__(self):
        return str(self)

cdef class Input(Node):
    """A Node designed to quickly read data from an input numpy array."""

    cdef double[:] data    # The input vector
    cdef uint64_t i, size  # Current index and max index
    cdef int64_t _cap      # Update cap at which to set output to zero and stop reading from the source. -1=uncapped
    cdef bint loop         # Whether to loop when i >= size

    property cap:
        "Update cap at which to set output to zero and stop reading from the source. A negative cap means no cap."
        def __get__(self):
            return int(self._cap)
        def __set__(self, int x):
            self._cap = <int64_t>x
        def __del__(self):
            self._cap = -1

    def __cinit__(self, double[:] data=None, object fn=None, double value=0.0, bint loop=False, int64_t cap=-1):
        self.is_input = 1

    def __init__(self, double[:] data=None, object fn=None, double value=0.0, bint loop=False, int64_t cap=-1):
        """Create a new Input object for feeding numerical data into a network.

        Keyword arguments:
        data : double[:] - The data vector; data is read sequentially from the vector at each update. Default is None.
        fn : str or callable - The APF; Input outputs fn(data) if fn is not None else (data). See Node.
        value : float - The initial potential of the node before any data is read. Default is zero.
        loop : bool - Whether to read data cyclically and continuously. Default is False; potential becomes zero when there is no more data.
        cap : int - The input cap. See the property description. Default is -1 (no cap).
        """
        if pthread_rwlock_init(&self.lock, NULL) != 0:
            raise RuntimeError('Can\'t initialize read/write lock.')
        self.pot[0] = value
        self.pot[1] = value
        self.fn = fn
        self.loop = loop
        self._cap = cap
        if data.size > 0:
            self.size = <uint64_t>data.size
            self.data = data

    # NOTE: does not throw exceptions.
    @boundscheck(False)
    cdef void _update(self, bint clock) nogil:
        if pthread_rwlock_wrlock(&self.lock) == 0:
            if self.data is not None and self._cap != 0:
                self._update_set(clock, self.data[self.i])
                self.i += 1
                if self.i >= self.size:
                    self.i = 0
                    if not self.loop:
                        self.data = None
                        self.size = 0
                if self._cap > 0:
                    self._cap -= 1
            else:
                self._update_set(clock, 0.0)
            pthread_rwlock_unlock(&self.lock)

    def __str__(self):
        return 'Input'

    @classmethod
    def Layer(self, double[:,:] data, object fn=None, double value=0.0, bint loop=False, int64_t cap=-1):
        """Returns a list of Input objects, each of which reads data from the corresponding column of a data matrix.

        Keyword arguments:
        data : double[:] - The data matrix; data is read sequentially down the rows at each update.
        fn : str or callable - The APF for each node; Layer outputs fn(data) if fn is not None else (data). See Node.
        value : float - The initial potential of each node before any data is read. Default is zero.
        loop : bool - Whether each node reads data cyclically and continuously. Default is False; potential becomes zero when there is no more data.
        cap : int - The input cap for each node. See the property description. Default is -1 (no cap).
        """
        cdef uint64_t i, cols = <uint64_t>data.shape[1]
        cdef list r = []
        for i in range(cols):
            r.append(self(data=data[:,i], fn=fn, value=value, loop=loop, cap=cap))
        return r

# Represents a parent (input) to a neuron.
cdef struct Parent:
    double weight   # The synaptic weight.
    PyObject *node  # The parent object.

cdef class Neuron(Node):
    """A Node designed to process input from other nodes and perform backpropagation."""

    cdef public double bias      # The node bias.
    cdef Parent *_parents        # Array of parents.
    cdef uint64_t c              # Number of parents.
    cdef double *dCdp            # Batch of partial derivatives (for backpropagation).
    cdef uint64_t _depth         # Depth from input layer (for backpropagation).

    def __cinit__(self, double bias=0.0, dict parents=None, object fn='sig', double value=0.0):
        self.is_neuron = 1

    def __init__(self, double bias=0.0, dict parents=None, object fn='sig', double value=0.0):
        """Create a new Neuron object for processing data in a network.

        Keyword arguments:
        bias : float - The node bias; Neuron value is (w * x + b) where w is synaptic weight, x is input and b is bias. Default is zero.
        parents : dict - Dictionary of the form {parent: weight} where parent is a Node object and weight is the associated synaptic weight. Default is None.
        fn : str or callable - The APF; see Node. Default is the logistic curve ('sig').
        value : float - The initial potential of the node before any data is processed. Default is zero.
        """
        if pthread_rwlock_init(&self.lock, NULL) != 0:
            raise RuntimeError('Can\'t initialize read/write lock.')
        self.pot[0] = value
        self.pot[1] = value
        self.fn = fn
        self.bias = bias
        self.connect(parents)

    # NOTE: doesn't even try to get the lock.
    def __dealloc__(self):
        if pthread_rwlock_wrlock(&self.lock) != 0:
            raise RuntimeError('Couldn\'t get write lock.')
        cdef uint64_t i
        try:
            for i in range(self.c):
                Py_DECREF(<object>self._parents[i].node)
            free(self._parents)
        finally:
            pthread_rwlock_unlock(&self.lock)

    def __len__(self):
        """Returns the number of parents."""
        cdef uint64_t r
        if pthread_rwlock_rdlock(&self.lock) != 0:
            raise RuntimeError('Couldn\'t get read lock.')
        r = self.c
        pthread_rwlock_unlock(&self.lock)
        return int(r)

    def __getitem__(self, object x):
        """Returns self's weight for parent x. Unconnected nodes are considered to have zero weight."""
        cdef uint64_t i
        cdef double r
        if pthread_rwlock_rdlock(&self.lock) != 0:
            raise RuntimeError('Couldn\'t get read lock.')
        try:
            i = self.index(<PyObject*>x)
            r = self._parents[i].weight if i < self.c else 0.0
        finally:
            pthread_rwlock_unlock(&self.lock)
        return float(r)

    # TODO: See if compound assignment covers this behavior already.
    def __setitem__(self, object x, double y):
        """Adds y to self's weight for x. Unconnected nodes as considered to have zero weight."""
        cdef uint64_t i
        if not isinstance(x, Node):
            raise TypeError('Parent must be a Node.')
        if pthread_rwlock_wrlock(&self.lock) != 0:
            raise RuntimeError('Couldn\'t get write lock.')
        try:
            i = self.index(<PyObject*>x)
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
        finally:
            pthread_rwlock_unlock(&self.lock)


    def __delitem__(self, object x):
        """Disconnects self from x."""
        cdef uint64_t i, j
        if pthread_rwlock_wrlock(&self.lock) != 0:
            raise RuntimeError('Couldn\'t get write lock.')
        try:
            i = self.index(<PyObject*>x)
            if i < self.c:
                Py_DECREF(<object>self._parents[i].node)
                self.c -= 1
                for j in range(i, self.c):
                    self._parents[j] = self._parents[j + 1]
                self._parents = <Parent*>realloc(self._parents, self.c * sizeof(Parent))
                if self._parents == NULL:
                    self.c = 0
                    raise MemoryError('Not enough memory to reallocate parents.')
        finally:
            pthread_rwlock_unlock(&self.lock)

    def __contains__(self, object x):
        """Returns True is x is connected to self."""
        cdef bint r
        if pthread_rwlock_rdlock(&self.lock) != 0:
            raise RuntimeError('Couldn\'t get read lock.')
        try:
            r = 1 if self.index(<PyObject*>x) < self.c else 0
        finally:
            pthread_rwlock_unlock(&self.lock)
        return bool(r)

    # Returns the index of node in parents else self.c.
    # Expects the caller to have a read/write lock!
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
        if pthread_rwlock_rdlock(&self.lock) != 0:
            raise RuntimeError('Couldn\'t get read lock.')
        try:
            for i in range(self.c):
                r[<object>self._parents[i].node] = float(self._parents[i].weight)
        finally:
            pthread_rwlock_unlock(&self.lock)
        return r

    def connect(self, dict parents=None):
        """Connects a mapping of {node: weight} parents to self."""
        cdef uint64_t l, i
        if pthread_rwlock_rdlock(&self.lock) != 0:
            raise RuntimeError('Couldn\'t get read lock.')
        try:
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
        finally:
            pthread_rwlock_unlock(&self.lock)

    # NOTE: does not raise exceptions!
    cdef void _update(self, bint clock) nogil:
        cdef double pot
        cdef uint64_t i
        if pthread_rwlock_wrlock(&self.lock) == 0:
            pot = self.bias
            for i in range(self.c):
                pot += (<Node>self._parents[i].node).pot[clock] * self._parents[i].weight
            self._update_set(clock, pot)
            pthread_rwlock_unlock(&self.lock)

    # Calculates the depth of the neuron from the input and output layers and checks for feedback loops.
    # Sets err to -1 if a feedback loop is found, -2 if the lock can't be attained, -3 if a memory error occurs. Returns zero on error.
    # NOTE: does not raise exceptions!
    cdef uint64_t depth(self, Path *path, int *err, bint force=False) nogil:
        cdef uint64_t i, cdepth, r = 0
        cdef Path *tmp
        err[0] = _calc_output_depth(<PyObject*>self, path)
        if err[0] == 0:
            if pthread_rwlock_wrlock(&self.lock) != 0:
                err[0] = -2
            else:
                if (<Neuron>self)._depth != 0 and not force:
                    r = (<Neuron>self)._depth
                else:
                    tmp = <Path*>malloc(sizeof(Path))
                    if tmp == NULL:
                        err[0] = -3
                    else:
                        tmp.child = <PyObject*>self
                        tmp.nxt = path
                        for i in range((<Neuron>self).c):
                            cdepth = (<Node>(<Neuron>self)._parents[i].node).depth(tmp, err, force)
                            if err[0] != 0:
                                break
                            if cdepth > r:
                                r = cdepth
                        free(tmp)
                        if err[0] == 0:
                            r += 1
                            (<Neuron>self)._depth = r
                        else:
                            r = 0
                pthread_rwlock_unlock(&self.lock)
        return r

    # Allocates and initializes the buffers used for backpropagation.
    # NOTE: doesn't even try to get the lock.
    cdef int _init_backprop(self) except -1:
        cdef uint64_t i, j = self.c + 1
        self.dCdp = <double*>realloc(self.dCdp, j * sizeof(double))
        if self.dCdp == NULL:
            raise MemoryError('Not enough memory to allocate self.dCdp.')
        for i in range(j):
            self.dCdp[i] = 0.0
        return 0

    # Performs parrallel recursive backpropagation on a batch. Calls the backpropation function of each of it's parents.
    # NOTE: does not throw exceptions. Returns -1 if the lock can't be attained.
    cdef int _backprop(self, double front, uint64_t output_depth) nogil:
        cdef uint64_t i
        cdef double dtmp
        cdef PyObject *tmp
        if pthread_rwlock_wrlock(&self.lock) != 0:
            return -1
        dtmp = self.pot_list[neg_mod(<int64_t>self.pl_i - <int64_t>output_depth, self.pl_n)] # past value
        front *= dtmp * (1.0 - dtmp)
        self.dCdp[self.c] += front  # dCdb
        pthread_rwlock_unlock(&self.lock)
        output_depth += 1
        for i in range(self.c):
            tmp = self._parents[i].node
            if pthread_rwlock_rdlock(&(<Node>tmp).lock) == 0:
                dtmp = (<Node>tmp).pot_list[neg_mod(<int64_t>(<Node>tmp).pl_i - <int64_t>output_depth, (<Node>tmp).pl_n)] * front
                pthread_rwlock_unlock(&(<Node>tmp).lock)
                if pthread_rwlock_wrlock(&self.lock) == 0:
                    self.dCdp[i] += dtmp
                    pthread_rwlock_unlock(&self.lock)
            if (<Node>tmp).is_neuron:
                (<Neuron>tmp)._backprop(self._parents[i].weight * front, output_depth)
        return 0

    # Tunes the neuron's parameters by averaging the partial derivatives calculated over a batch.
    # NOTE: does not throw exceptions.
    cdef void _register_backprop(self, double alpha, double lamb) nogil:
        cdef uint64_t i
        if pthread_rwlock_wrlock(&self.lock) == 0:
            self.bias -= alpha * self.dCdp[self.c]
            self.dCdp[self.c] = 0.0
            for i in range(self.c):
                self._parents[i].weight -= alpha * (self.dCdp[i] + lamb * self._parents[i].weight)
                self.dCdp[i] = 0.0
            pthread_rwlock_unlock(&self.lock)

    # Deallocates the buffers used in backpropagation.
    cdef void _dealloc_backprop(self) nogil:
        if pthread_rwlock_wrlock(&self.lock) == 0:
            free(self.dCdp)
            self.dCdp = NULL
            pthread_rwlock_unlock(&self.lock)

    def __str__(self):
        return 'Neuron(degree=%d, bias=%f)' % (self.c, self.bias)

cdef struct bp_args:
    pthread_t thread
    PyObject *node
    double front
    bint is_inited

# Start routine for recursive packpropagation.
cdef void *start_backprop(void *args) nogil:
    (<Neuron>(<bp_args*>args).node)._backprop((<bp_args*>args).front, 1)
    return NULL

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
        #for i in prange(self.c, nogil=True):
        for i in range(self.c):
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
                raise MemoryError('Couldn\'t get the lock.')
            if err == -3:
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

    cdef int _bp_init(self, uint64_t times, bint verbose=False) except -1:
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
            if (<Node>self._nodes[i]).is_input:
                (<Input>self._nodes[i])._cap = <int64_t>times
        if verbose:
            print('Done.')
        return 0

    cdef void _bp_prerun(self, uint64_t depth, bint verbose=False):
        if verbose:
            print('Pre-running...'),
        cdef uint64_t i
        for i in range(depth):
            self._update_once(NULL, 0)
        if verbose:
            print('Done.')

    cdef void _bp_dealloc(self, bint verbose=False):
        if verbose:
            print('Freeing backpropagation buffers ...'),
        cdef uint64_t i
        #for i in prange(self.c, nogil=True):
        for i in range(self.c):
            if (<Node>self._nodes[i]).is_neuron:
                (<Neuron>self._nodes[i])._dealloc_backprop()
            if (<Node>self._nodes[i]).is_input:
                (<Input>self._nodes[i])._cap = -1
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
        cdef bp_args *args = <bp_args*>malloc(self.oc * sizeof(bp_args))
        if args == NULL:
            raise MemoryError('Not enough memory to allocate bp_args.')
        cdef uint64_t i, j, k
        for i in range(self.oc):
            args[i].is_inited = False
        depth = self._bp_depth(depth, verbose)
        alpha /= <double>batch
        #lamb *= <double>batch
        cdef uint64_t l = expect.size / (self.oc * batch)
        self._bp_init(l * batch, verbose)
        self._bp_prerun(depth, verbose)
        if verbose:
            print('Running [%d] {' % l)
        cdef double cost, c
        cdef double g, e, max_g, max_e
        cdef uint64_t max_g_i, max_e_i, correct = 0
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
                        args[k].front = c
                        args[k].node = self._output[k]
                        if args[k].is_inited:
                            if pthread_join(args[k].thread, NULL) != 0:
                                pass  # TODO: handle error
                        with nogil:
                            if pthread_create(&args[k].thread, NULL, &start_backprop, &args[k]) != 0:
                                pass  # TODO: handle error
                        args[k].is_inited = True
                    cost += c * c
                if max_g_i == max_e_i:
                    correct += 1
            if verbose:
                print('\tBatch [%d] Cost = %f' % (batch, cost / (2.0 * batch)))
            for k in range(self.oc):
                if pthread_join(args[k].thread, NULL) != 0:
                    pass  # TODO: handle error
                args[k].is_inited = False
            for k in range(self.c):
                if (<Node>self._nodes[k]).is_neuron:
                    (<Neuron>self._nodes[k])._register_backprop(alpha, lamb)
        if verbose:
            print('} Done')
            print('Accuracy = %f' % (<double>correct / <double>(batch * l)))
        free(args)
        self._bp_dealloc(verbose)

    cdef uint64_t index(self, PyObject *node):
        """Returns the index of node in _nodes else self.c."""
        cdef uint64_t i
        for i in range(self.c):
            if self._nodes[i] == node:
                return i
        return self.c

    cdef uint64_t oindex(self, PyObject *node):
        """Returns the index of node in _output else self.oc."""
        cdef uint64_t i
        for i in range(self.oc):
            if self._output[i] == node:
                return i
        return self.oc

    def write(self, filename): # TODO: write
        """Encodes, compresses, and saves the network structure to file."""
        f = open(filename, 'wb')
        f.write(pack('!QQ', self.c, self.oc))
        for i in range(self.c):
            if (<Node>self._nodes[i]).is_neuron:
                rents = (<Neuron>self._nodes[i]).parents().items()
                l = len(rents)
                f.write(pack('!BdQ', 2, (<Neuron>self._nodes[i]).bias, l))
                for i in range(l):
                    n, w = rents[i]
                    f.write(pack('!Qd', self.index(<PyObject*>n), float(w)))
            elif (<Node>self._nodes[i]).is_input:
                f.write(pack('!B', 1))
            else:
                f.write(pack('!B', 0))
            f.write(pack('!Q', self.oindex(self._nodes[i])))
        f.close()

    @classmethod
    def open(self, filename, double[:,:] data, bint loop=0, int64_t cap=-1): # TODO: write
        """Reads, decompresses, and decodes a network structure from a file."""
        f = open(filename, 'rb')
        c, oc = unpack('!QQ', f.read(16))
        nodes = []
        rents = []
        output = [None] * oc
        cdef uint64_t input_i = 0
        for i in range(c):
            t, = unpack('!B', f.read(1))
            if t == 2:    # Neuron
                b, n = unpack('!dQ', f.read(16))
                r = []
                for j in range(n):
                    p, w = unpack('!Qd', f.read(16))
                    r.append((p, w))
                nodes.append(Neuron(bias=b, fn='sig'))
                rents.append(r)
            elif t == 1:  # Input
                nodes.append(Input(data=data[:,input_i], loop=loop, cap=cap))
                input_i += 1
                rents.append(None)
            else:         # Node
                nodes.append(Node())
                rents.append(None)
            oind, = unpack('!Q', f.read(8))
            if oind < oc:
                output[oind] = nodes[-1]
        for i in range(c):
            if rents[i] is not None:
                nodes[i].connect({nodes[ind]: w for ind, w in rents[i]})
        return Network(nodes=nodes, output=[o for o in output if o is not None])

    def __str__(self):
        cdef uint64_t i
        r = 'Network {\n'
        for i in range(self.c):
            r += '    ' + str(<object>self._nodes[i]) + '\n'
        return r + '}'

    def __repr__(self):
        return str(self)

    @classmethod
    def Layered(self, object layers, double[:,:] data, bint loop=0, int64_t cap=-1):
        """Returns a new Network object in a standard layered feed-forward structure.

        Keyword arguments:
        layers -- A list of layer widths. width i is the number of neurons in layer i from the input.
        data -- The input data to the network. An input layer is created using Input.Layer(data=data, loop=loop)
        loop -- Whether the Inputs read data cyclically and continuously. By default, potential becomes zero when there is no more data.
        """
        nodes = Input.Layer(data=data, loop=loop, cap=cap)
        last_w = len(nodes)
        if last_w == 0:
            raise ValueError('Input data must not be empty.')
        for w in layers:
            if w <= 0:
                raise ValueError('Layer width must be positive.')
            nodes.extend([Neuron(bias=normal(), parents={n: normal(0.0, 1.0/sqrt(last_w)) for n in nodes[-last_w:]}, fn='sig') for i in range(w)])
            last_w = w
        return self(nodes=nodes, output=nodes[-last_w:])

