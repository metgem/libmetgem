# cython: language_level=3

from libcpp.vector cimport vector
cimport numpy as np
from ._common cimport peak_t

cdef vector[peak_t] filter_data_nogil(double, const peak_t *, np.npy_intp, int, int, int, int) nogil