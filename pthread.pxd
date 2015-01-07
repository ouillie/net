# pthread.pxd: Cython definitions for libpthread.

cdef extern from 'pthread.h':
    ctypedef long pthread_t              # Place-holder
    extern struct pthread_attr_t:        # Place-holder
        pass
    ctypedef long pthread_rwlock_t       # Place-holder
    extern struct pthread_rwlockattr_t:  # Place-holder
        pass
    extern int pthread_create(pthread_t*, const pthread_attr_t*, void *(*)(void*), void*) nogil
    extern int pthread_join(pthread_t, void**) nogil
    extern int pthread_rwlock_rdlock(pthread_rwlock_t*) nogil
    extern int pthread_rwlock_wrlock(pthread_rwlock_t*) nogil
    extern int pthread_rwlock_unlock(pthread_rwlock_t*) nogil
    extern int pthread_rwlock_init(pthread_rwlock_t*, pthread_rwlockattr_t*) nogil

