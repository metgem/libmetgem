# cython: language_level=3

cimport numpy as np

from ._common cimport peak_t, score_algorithm_t

cpdef enum SpectraMatchState:
    fragment = 0
    neutral_loss = 1

cdef double generic_score_nogil(
    double, const peak_t*, np.npy_intp,
    double, const peak_t*, np.npy_intp,
    double, int, score_algorithm_t score_algorithm=*) noexcept nogil