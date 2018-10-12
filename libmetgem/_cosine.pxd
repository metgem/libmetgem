from ._common cimport peak_t

cdef double cosine_score_nogil(double, peak_t *, int, double, peak_t *, int, double, int) nogil