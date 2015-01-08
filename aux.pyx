# aux.pyx: Cython definitions for miscellaneous C functions and types

from libc.math cimport log
from cython cimport boundscheck

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
def step(double t):
    """Returns 1 if t > 0 else 0."""
    return float(_bin(t))

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

# Built-in cost functions.
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

