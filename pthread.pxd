
cdef extern from 'pthread.h':
    ctypedef int pthread_t                 # Place-holder
    ctypedef struct pthread_attr_t:        # Place-holder
        pass
    ctypedef struct pthread_rwlock_t:      # Place-holder
        pass
    ctypedef struct pthread_rwlockattr_t:  # Place-holder
        pass
    int pthread_create(pthread_t*, const pthread_attr_t*, void *(*)(void *), void *) nogil
    int pthread_rwlock_init(pthread_rwlock_t*, const pthread_rwlockattr_t*) nogil
    int pthread_rwlock_rdlock(pthread_rwlock_t*) nogil
    int pthread_rwlock_wrlock(pthread_rwlock_t*) nogil
    int pthread_rwlock_unlock(pthread_rwlock_t*) nogil

