# cython: language_level=3

from libcpp.vector cimport vector
from libcpp cimport bool
cimport numpy as np
from ._common cimport peak_t, norm_method_t

cdef vector[peak_t] filter_data_nogil(double, const peak_t *, np.npy_intp, int, int, int, int, double, bool square_root=*, norm_method_t norm_method=*) noexcept nogil
cdef vector[peak_t] normalize_data_nogil(vector[peak_t], norm_method_t norm_method=*) noexcept nogil