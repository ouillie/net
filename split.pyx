

from libc.stdlib cimport malloc, free
cimport numpy as np

cdef shqueue *new_shqueue(np.ndarray[double] seq) except NULL:
    cdef uint64_t size = seq.size
    if size == 0:
        raise ValueError('Generator must yield non-empty numpy arrays.')
        return NULL
    cdef shqueue *r = <shqueue*>malloc(sizeof(shqueue))
    if r == NULL:
        raise MemoryError('Not enough memory to allocate shqueue.')
        return NULL
    r.vector = <double*>malloc(size * sizeof(double))
    if r.vector == NULL:
        free(r)
        raise MemoryError('Not enough memory to allocate shqueue.vector.')
        return NULL
    cdef size_t bm_len = (size - 1) / sizeof(unsigned int) + 1
    r.spent = <unsigned int*>malloc(bm_len * sizeof(unsigned int))
    if r.spent == NULL:
        free(r.vector)
        free(r)
        raise MemoryError('Not enough memory to allocate shqueue.spent.')
        return NULL
    cdef uint64_t i
    for i in range(bm_len):
        r.spent[i] = 0
    for i in range(size):
        r.vector[i] = seq[i]
    r.nxt = NULL
    return r

cdef void free_shqueue(shqueue *shq) nogil:
    free(shq.vector)
    free(shq.spent)
    free(shq)

cdef class Splitter:

    def __cinit__(self, SplitterHouse house, uint64_t i):
        self.house = house
        self.i = i

    cdef double nxt(self, int *err) nogil:
        cdef double r = 0.0
        cdef shqueue *shq
        if self.house is None:
            if err != NULL:
                err[0] = -1
        else:
            shq = self.house.first
            while shq != NULL and shq.spent[self.i/sizeof(unsigned int)] & (1 << self.i % sizeof(unsigned int)) != 0:
                shq = shq.nxt
            if shq == NULL:
                with gil:
                    if self.house.get_next() == -1:
                        self.house = None
                        if err != NULL:
                            err[0] = -1
                        return r
                shq = self.house.last
            shq.spent[self.i/sizeof(unsigned int)] |= (1 << self.i % sizeof(unsigned int))
            r = shq.vector[self.i]
            self.house.purge()
        return r

    def __next__(self):
        cdef int err = 0
        cdef double r = self.nxt(&err)
        if err == -1:
            raise StopIteration
        return r

cdef class SplitterHouse:

    def __cinit__(self, object source):
        cdef np.ndarray[double] head
        try:
            head = next(source)
        except StopIteration:
            pass
        else:
            self.width = head.size
            self.first = new_shqueue(head)
            self.last = self.first
            self.source = source

    def __dealloc__(self):
        cdef shqueue *shq = self.first
        cdef shqueue *nxt
        while shq != NULL:
            nxt = shq.nxt
            free_shqueue(shq)
            shq = nxt

    cdef int get_next(self) except -1:
        if self.source is None:
            raise StopIteration
            return -1
        cdef np.ndarray[double] n
        try:
            n = next(self.source)
        except:
            self.source = None
            raise
            return -1
        cdef shqueue *shq = new_shqueue(n)
        if self.first == NULL:
            self.first = shq
        if self.last != NULL:
            self.last.nxt = shq
        self.last = shq
        return 0

    cdef void purge(self) nogil:
        cdef shqueue *shq = self.first
        cdef shqueue *nxt = NULL
        cdef shqueue *prev = NULL
        cdef uint64_t i
        cdef bint should_free
        while shq != NULL:
            should_free = 1
            for i in range(self.width):
                if shq.spent[i/sizeof(unsigned int)] & (1 << i % sizeof(unsigned int)) == 0:
                    should_free = 0
                    break
            if should_free:
                nxt = shq.nxt
                free_shqueue(shq)
                if shq.nxt == NULL:
                    self.last = prev
                if prev == NULL:
                    self.first = shq.nxt
                else:
                    prev.nxt = shq.nxt
            prev = shq
            shq = shq.nxt

