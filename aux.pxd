# aux.pxd: Cython declarations for miscellaneous C functions and types

cdef extern from 'stdint.h':
    ctypedef unsigned long long uint64_t
    ctypedef signed long long int64_t

cdef extern from 'cfns.h':
    ctypedef double (*nodefn)(double) nogil
    extern double _sig(double) nogil
    extern double _bin(double) nogil

cdef extern uint64_t neg_mod(int64_t, uint64_t) nogil

cdef extern object sig(double)
cdef extern object step(double)

cdef extern nodefn cfns(str)
cdef extern str re_cfns(nodefn)

cdef extern object quad_cost(double[:], double[:])
cdef extern object cent_cost(double[:], double[:])

