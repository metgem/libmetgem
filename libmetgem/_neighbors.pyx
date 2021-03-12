# cython: language_level=3
# distutils: language=c++

cimport cython
import numpy as np
from scipy.sparse import csr_matrix
cimport numpy as np
from libcpp.vector cimport vector
from cython cimport integral
from cython.parallel cimport prange, parallel

np.import_array()
    
cdef extern from "argpartition.h" nogil:
   vector[T] argpartition[T](vector[float] vec, const int &N)
   
cdef NUMPY_TYPE_MAP = {2 : np.int16, 4: np.int32, 8: np.int64}

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _kneighbors_graph_from_similarity_matrix_nogil(
        np.float32_t[:] data, integral[:] indices, integral[:] indptr,
        int n_samples, int n_neighbors):
    cdef:
        integral n_nonzero = n_samples * (n_neighbors+1)
        np.float32_t[:] r_data = np.empty(n_nonzero, dtype=np.float32)
        integral[:] r_indices = np.empty(n_nonzero, dtype=NUMPY_TYPE_MAP[sizeof(indices[0])])
        integral[:] r_indptr = np.arange(0, n_nonzero + 1, n_neighbors+1, dtype=NUMPY_TYPE_MAP[sizeof(indices[0])])
        vector[np.float32_t] row
        vector[integral] ind
        vector[integral] inds
        integral r_start, r_end, index
        integral i, j
    
    with nogil, parallel():
        row = vector[np.float32_t](n_samples, 1.)
        inds = vector[integral](n_samples)
        
        for i in prange(n_samples, schedule='guided'):           
            r_start = indptr[i]
            r_end = indptr[i+1]
            for j, index in enumerate(range(r_start, r_end)):
                inds[j] = indices[index]
                row[inds[j]] = 1. - data[index]
                
            ind = argpartition[cython.typeof(n_nonzero)](row, n_neighbors)

            for j in range(n_neighbors+1):
                r_data[r_indptr[i]+j] = row[ind[j]]
                r_indices[r_indptr[i]+j] = ind[j]
                
            for j in range(r_end-r_start):
                row[inds[j]] = 1.
                
    return csr_matrix((r_data, r_indices, r_indptr))

def kneighbors_graph_from_similarity_matrix(matrix: csr_matrix, int n_neighbors):
    cdef:
        int n_samples = matrix.shape[0]

    if n_neighbors > n_samples:
        raise ValueError(f"Expected n_neighbors <= n_samples,  but n_samples = {n_samples}, n_neighbors = {n_neighbors}")
    elif n_neighbors == n_samples:
        return matrix

    if matrix.indices.dtype == np.int64:
        return _kneighbors_graph_from_similarity_matrix_nogil[long](matrix.data,
                                                                    matrix.indices,
                                                                    matrix.indptr,
                                                                    n_samples, n_neighbors)
    elif matrix.indices.dtype == np.int16:
        return _kneighbors_graph_from_similarity_matrix_nogil[short](matrix.data,
                                                                    matrix.indices,
                                                                    matrix.indptr,
                                                                    n_samples, n_neighbors)
    else:
        return _kneighbors_graph_from_similarity_matrix_nogil[int](matrix.data,
                                                                   matrix.indices,
                                                                   matrix.indptr,
                                                                   n_samples, n_neighbors)
