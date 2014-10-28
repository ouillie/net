
cdef extern from 'stdint.h':
    ctypedef unsigned long long uint64_t

cdef struct shqueue:
    double *vector
    unsigned int *spent  # arbitrary-length bit-mask of which items in the vector have been visited (1=visited)
    shqueue *nxt

cdef class Splitter:
    cdef SplitterHouse house  # it's house
    cdef uint64_t i           # the index of this splitter in it's house
    cdef double nxt(self, int *err) nogil

cdef class SplitterHouse:
    cdef object source
    cdef shqueue *first
    cdef shqueue *last
    cdef uint64_t width, ref
    cdef int get_next(self) except -1
    cdef void purge(self) nogil

