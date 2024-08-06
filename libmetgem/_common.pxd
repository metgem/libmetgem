# cython: language_level=3

cimport numpy as np
from libcpp.vector cimport vector
from libc.stdint cimport uint16_t, uint8_t
from cython cimport numeric

ctypedef struct peak_t:
    float mz
    float intensity
    
ctypedef packed struct score_t:
    uint16_t ix1
    uint16_t ix2
    double value
    uint8_t type
    
ctypedef enum score_algorithm_t:
    cosine
    entropy
    weighted_entropy
    
ctypedef enum norm_method_t:
    dot
    sum
    
cdef np.ndarray[np.float32_t, ndim=2] arr_from_peaks_vector(vector[peak_t] v)
cdef np.ndarray arr_from_score_vector(vector[score_t] v)
cdef np.ndarray arr_from_1d_vector(vector[numeric] v, np.dtype dtype)
cdef void *np_arr_pointer(np.ndarray[numeric, ndim=2] data)
cdef score_algorithm_t str_to_score_algorithm(str)
cdef norm_method_t str_to_norm_method(str)