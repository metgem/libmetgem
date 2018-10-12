from libcpp.vector cimport vector
from ._common cimport peak_t

cdef vector[peak_t] filter_data_nogil(double, const peak_t *, int, int, int, int, int) nogil