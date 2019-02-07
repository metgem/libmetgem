# cython: language_level=3

cimport numpy as np
from libcpp.vector cimport vector

ctypedef struct peak_t:
    float mz
    float intensity
    
cdef float[:,:] arr_from_vector(vector[peak_t])
cdef peak_t *np_arr_pointer(np.ndarray[np.float32_t, ndim=2] data)