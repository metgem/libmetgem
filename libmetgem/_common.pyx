# cython: language_level=3
# distutils: language=c++

import numpy as np
cimport numpy as np
cimport cython
from cython.view cimport array as cvarray
from libcpp.vector cimport vector
from cython.parallel cimport prange
from libc.stdlib cimport free, malloc
from cpython cimport PyObject, Py_INCREF, Py_DECREF
from cpython.ref cimport PyTypeObject

cdef extern from "<numpy/arrayobject.h>":
    object PyArray_NewFromDescr(PyTypeObject *subtype, np.dtype descr, int nd,
                                np.npy_intp* dims, np.npy_intp* strides,
                                void* data, int flags, object obj)

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class ArrayWrapper:
    cdef:
        np.dtype dtype
        tuple shape
        void *ptr
        
    def __init__(self, tuple shape, np.dtype dtype):
        self.shape = shape
        self.dtype = dtype
        Py_INCREF(dtype)

    cdef set_data(self, void *ptr):
        self.ptr = ptr
        
    def __array__(self):
        cdef np.ndarray arr
        cdef Py_ssize_t size = len(self.shape)
        cdef np.npy_intp* shape
                    
        if self.dtype.fields:
            size -= 1
            
        shape = <np.npy_intp *> malloc(size * sizeof(np.npy_intp))
        
        for i in range(size):
            shape[i] = self.shape[i]
            
        arr = PyArray_NewFromDescr(<PyTypeObject *> np.ndarray, self.dtype,
                                       <int> size, shape, NULL, self.ptr,
                                       np.NPY_DEFAULT, <object> NULL)
        free(<void*>shape)
        Py_INCREF(self)
        return arr
        
    def __dealloc__(self):
        self.ptr = NULL
        Py_DECREF(self.dtype)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float32_t, ndim=2] arr_from_peaks_vector(vector[peak_t] v):
    if v.empty():
        return np.empty((0, 2), dtype=np.float32)
    array_wrapper = ArrayWrapper((v.size(), 2), np.dtype(np.float32))
    array_wrapper.set_data(v.data())    
    return np.array(array_wrapper)

@cython.boundscheck(False)
@cython.wraparound(False)   
cdef np.ndarray arr_from_score_vector(vector[score_t] v):
    cdef np.dtype dtype = np.dtype([('ix1', '<u2'), ('ix2', '<u2'), ('score', '<f8'), ('type', '<u1')])
    if v.empty():
        return np.empty((0,1), dtype=dtype)
    array_wrapper = ArrayWrapper((v.size(), 3), dtype)
    array_wrapper.set_data(v.data())    
    return np.array(array_wrapper)

@cython.boundscheck(False)
@cython.wraparound(False)   
cdef np.ndarray arr_from_1d_vector(vector[numeric] v, np.dtype dtype):
    if v.empty():
        return np.empty((0, 1), dtype=dtype)
    array_wrapper = ArrayWrapper((v.size(),), dtype)
    array_wrapper.set_data(v.data())    
    return np.array(array_wrapper)
   
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void *np_arr_pointer(np.ndarray[numeric, ndim=2] data):
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data, dtype=data.dtype)
    return <void *>&data[0, 0]
