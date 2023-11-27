# cython: language_level=3

cimport numpy as np

from ._common cimport peak_t

cpdef enum SpectraMatchState:
    fragment = 0
    neutral_loss = 1

cdef double cosine_score_nogil(double, peak_t*, np.npy_intp,
                               double, peak_t*, np.npy_intp,
                               double, int) noexcept nogil