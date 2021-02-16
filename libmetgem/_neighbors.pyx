# cython: language_level=3
# distutils: language=c++

import numpy as np
from scipy.sparse import csr_matrix
cimport numpy as np
cimport cython
from libcpp.vector cimport vector
from cython cimport numeric
from cython.parallel import prange
from libcpp cimport bool
from libcpp.algorithm cimport partial_sort

from ._common cimport np_arr_pointer_1d

np.import_array()
    
cdef extern from "argpartition.h" nogil:
   vector[int] argpartition(vector[float] vec, const int &N)


cdef void _kneighbors_graph_from_similarity_matrix_nogil(
        np.float32_t *m_data, int *m_indices, int *m_indptr,
        int n_samples, int n_neighbors,
        np.float32_t *r_data, int *r_indices, int *r_indptr) nogil:
    cdef:
        vector[np.float32_t] row = vector[np.float32_t](n_samples, 1.)
        vector[int] ind = vector[int](n_neighbors)
        vector[int] inds = vector[int](n_samples)
        int r_start, r_end, index
        int i, j

    for i in prange(n_samples):
        r_start = m_indptr[i]
        r_end = m_indptr[i+1]
        for j, index in enumerate(range(r_start, r_end)):
            inds[j] = m_indices[index]
            row[inds[j]] = 1. - m_data[index]
            
        ind = argpartition(row, n_neighbors)

        for j in range(n_neighbors+1):
            r_data[r_indptr[i]+j] = row[ind[j]]
            r_indices[r_indptr[i]+j] = ind[j]
            
        for j in range(r_end-r_start):
            row[inds[j]] = 1.


def kneighbors_graph_from_similarity_matrix(matrix: csr_matrix, int n_neighbors):
    cdef:
        int n_samples = matrix.shape[0]
        int n_nonzero = n_samples * (n_neighbors+1)
        np.ndarray[np.float32_t, ndim=1] m_data = matrix.data
        np.ndarray[np.float32_t, ndim=1] r_data = np.empty(n_nonzero, dtype=np.float32)
        np.ndarray[int, ndim=1] m_indices = matrix.indices
        np.ndarray[int, ndim=1] r_indices = np.empty(n_nonzero, dtype=np.int32)
        np.ndarray[int, ndim=1] m_indptr = matrix.indptr
        np.ndarray[int, ndim=1] r_indptr = np.arange(0, n_nonzero + 1, n_neighbors+1, dtype=np.int32)
        
    if n_neighbors > n_samples:
        raise ValueError("Expected n_neighbors <= n_samples,  but n_samples = {n_samples}, n_neighbors = {n_neighbors}")
    elif n_neighbors == n_samples:
        return matrix
    
    _kneighbors_graph_from_similarity_matrix_nogil(
        np_arr_pointer_1d(m_data),
        np_arr_pointer_1d(m_indices),
        np_arr_pointer_1d(m_indptr),
        n_samples, n_neighbors,
        np_arr_pointer_1d(r_data),
        np_arr_pointer_1d(r_indices),
        np_arr_pointer_1d(r_indptr))
    return csr_matrix((r_data, r_indices, r_indptr))