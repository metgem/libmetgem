# cython: language_level=3
# distutils: language=c++

import warnings

cimport cython
cimport openmp
from cython.parallel import prange
from cython.view cimport array as cvarray
import numpy as np
cimport numpy as np
from libcpp cimport bool
from libcpp.algorithm cimport sort
from libcpp.vector cimport vector
from libc.math cimport fabs
from libc.stdint cimport uint16_t, uint8_t
from scipy.sparse import csr_matrix

cdef extern from "<numeric>" namespace "std" nogil:
    T accumulate[Iter, T](Iter first, Iter last, T init)
    T accumulate[Iter, T, BinaryOperation](Iter first, Iter last, T init, BinaryOperation op) except +

from ._common cimport (score_t, np_arr_pointer,
                       arr_from_score_vector,
                       arr_from_1d_vector)
                       
cdef double addScore(double sum, const score_t &b) noexcept nogil:
    return sum + b.value
  
cdef bool compareByScore(const score_t &a, const score_t &b) noexcept nogil:
    return a.value > b.value


cdef vector[score_t] compare_spectra_nogil(double spectrum1_mz,
                                           peak_t *spectrum1_data,
                                           np.npy_intp spectrum1_size,
                                           double spectrum2_mz,
                                           peak_t *spectrum2_data,
                                           np.npy_intp spectrum2_size,
                                           double mz_tolerance) noexcept nogil:
    cdef:
        double dm
        vector[score_t] scores
        vector[score_t] matches
        score_t pscore
        float diff
        int i, j
        unsigned int ix1, ix2
        vector[bool] peak_used1
        vector[bool] peak_used2
    
    if spectrum1_size == 0 or spectrum2_size == 0:
        return matches
    
    dm = spectrum1_mz - spectrum2_mz

    scores.reserve(spectrum1_size * spectrum2_size)
    if dm == 0.:
        for i in range(spectrum1_size):
            for j in range(spectrum2_size):
                diff = spectrum2_data[j].mz - spectrum1_data[i].mz
                if fabs(diff) <= mz_tolerance:
                    pscore.ix1 = i
                    pscore.ix2 = j
                    pscore.value = spectrum1_data[i].intensity * spectrum2_data[j].intensity
                    pscore.type = SpectraMatchState.fragment
                    scores.push_back(pscore)
    else:
        for i in range(spectrum1_size):
            for j in range(spectrum2_size):
                diff = spectrum2_data[j].mz - spectrum1_data[i].mz
                if fabs(diff) <= mz_tolerance:
                    pscore.ix1 = i
                    pscore.ix2 = j
                    pscore.value = spectrum1_data[i].intensity * spectrum2_data[j].intensity
                    pscore.type = SpectraMatchState.fragment
                    scores.push_back(pscore)
                elif fabs(diff + dm) <= mz_tolerance:
                    pscore.ix1 = i
                    pscore.ix2 = j
                    pscore.value = spectrum1_data[i].intensity * spectrum2_data[j].intensity
                    pscore.type = SpectraMatchState.neutral_loss
                    scores.push_back(pscore)
    
    if scores.size() == 0:
        return matches
    
    sort(scores.begin(), scores.end(), &compareByScore)
    
    peak_used1.resize(spectrum1_size)
    peak_used2.resize(spectrum2_size)
    
    matches.reserve(spectrum1_size * spectrum2_size)
    for i in range(scores.size()):
        ix1 = scores[i].ix1
        ix2 = scores[i].ix2
        if not peak_used1[ix1] and not peak_used2[ix2]:
            peak_used1[ix1] = 1
            peak_used2[ix2] = 1
            matches.push_back(scores[i])
            
    # Free memory
    matches.shrink_to_fit()
    scores.clear()
    scores.shrink_to_fit()
    peak_used1.clear()
    peak_used1.shrink_to_fit()
    peak_used2.clear()
    peak_used2.shrink_to_fit()

    return matches

cdef double cosine_score_nogil(double spectrum1_mz,
                               peak_t *spectrum1_data,
                               np.npy_intp spectrum1_size,
                               double spectrum2_mz,
                               peak_t *spectrum2_data,
                               np.npy_intp spectrum2_size,
                               double mz_tolerance,
                               int min_matched_peaks) noexcept nogil:
    cdef:
        vector[score_t] matches = compare_spectra_nogil(spectrum1_mz,
                                                        spectrum1_data,
                                                        spectrum1_size,
                                                        spectrum2_mz,
                                                        spectrum2_data,
                                                        spectrum2_size,
                                                        mz_tolerance)      
                
    if matches.size() < min_matched_peaks:
        return 0.

    try:
        return accumulate(matches.begin(), matches.end(), 0., &addScore)
    finally:
        matches.clear()
        matches.shrink_to_fit()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float[:,:] compute_similarity_matrix_dense_nogil(vector[double] mzvec,
                                                      vector[peak_t *] datavec,
                                                      vector[np.npy_intp] data_sizes,
                                                      double mz_tolerance,
                                                      int min_matched_peaks,
                                                      object callback=None):
    cdef:
        int i, j
        size_t size = mzvec.size()
        float[:,:] matrix = cvarray(shape=(size, size), itemsize=sizeof(float), format='f')
        bool has_callback = callback is not None
    
    with nogil:
        for i in prange(<int>size, schedule='guided'):
            matrix[i, i] = 1
            for j in range(i):
                matrix[i, j] = matrix[j, i] = <float> cosine_score_nogil(mzvec[i], datavec[i], data_sizes[i],
                                                                        mzvec[j], datavec[j], data_sizes[j],
                                                                        mz_tolerance, min_matched_peaks)
            if has_callback:
                with gil:
                    if not callback(i):
                        return matrix
    
    return matrix
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef (vector[float], vector[int], vector[int]) compute_similarity_matrix_sparse_nogil(
           vector[double] mzvec,
           vector[peak_t *] datavec,
           vector[np.npy_intp] data_sizes,
           double mz_tolerance,
           int min_matched_peaks,
           object callback=None):
    cdef:
        int i, j
        size_t size = mzvec.size()
        vector[float] data
        vector[int] col_ind
        vector[int] row_ind
        float score
        bool has_callback = callback is not None
        openmp.omp_lock_t lock
    
    openmp.omp_init_lock(&lock)
    
    with nogil:
        for i in prange(<int>size, schedule='guided'):
            openmp.omp_set_lock(&lock)
            data.push_back(1)
            col_ind.push_back(i)
            row_ind.push_back(i)
            openmp.omp_unset_lock(&lock)
            for j in range(i):
                score = <float> cosine_score_nogil(mzvec[i], datavec[i], data_sizes[i],
                                                   mzvec[j], datavec[j], data_sizes[j],
                                                   mz_tolerance, min_matched_peaks)
                if score > 0:
                    score = min(score, 1.)
                    openmp.omp_set_lock(&lock)
                    # upper triangle
                    data.push_back(score)
                    col_ind.push_back(i)
                    row_ind.push_back(j)
                    # lower triangle
                    data.push_back(score)
                    col_ind.push_back(j)
                    row_ind.push_back(i)
                    openmp.omp_unset_lock(&lock)
            if has_callback:
                with gil:
                    if not callback(i):
                        return data, row_ind, col_ind
    
    return data, row_ind, col_ind

def cosine_score(double spectrum1_mz,
                 np.ndarray[np.float32_t, ndim=2] spectrum1_data,
                 double spectrum2_mz,
                 np.ndarray[np.float32_t, ndim=2] spectrum2_data,
                 double mz_tolerance,
                 int min_matched_peaks):   
    return cosine_score_nogil(spectrum1_mz, <peak_t*>np_arr_pointer(spectrum1_data),
                              spectrum1_data.shape[0],
                              spectrum2_mz, <peak_t*>np_arr_pointer(spectrum2_data),
                              spectrum2_data.shape[0],
                              mz_tolerance, min_matched_peaks)
      
def compare_spectra(double spectrum1_mz,
                    np.ndarray[np.float32_t, ndim=2] spectrum1_data,
                    double spectrum2_mz,
                    np.ndarray[np.float32_t, ndim=2] spectrum2_data,
                    double mz_tolerance):
    cdef vector[score_t] matches
    matches = compare_spectra_nogil(spectrum1_mz,
                                    <peak_t*>np_arr_pointer(spectrum1_data),
                                    spectrum1_data.shape[0],
                                    spectrum2_mz,
                                    <peak_t*>np_arr_pointer(spectrum2_data),
                                    spectrum2_data.shape[0],
                                    mz_tolerance)
    return np.asarray(arr_from_score_vector(matches))

def compute_similarity_matrix(vector[double] mzvec, list datavec,
                              double mz_tolerance, int min_matched_peaks,
                              object callback=None, dense_output=True):
    cdef:
        np.ndarray[np.float32_t, ndim=2] tmp_array
        size_t size = mzvec.size()
        int i
        vector[peak_t *] data_p
        vector[np.npy_intp] data_sizes
        
    data_p.resize(size)
    data_sizes.resize(size)
    for i, tmp_array in enumerate(datavec):
        data_p[i] = <peak_t*>np_arr_pointer(tmp_array)
        data_sizes[i] = tmp_array.shape[0]
        
    if dense_output:
        matrix = np.asarray(compute_similarity_matrix_dense_nogil(mzvec, data_p, data_sizes, mz_tolerance, min_matched_peaks, callback))
        matrix[matrix>1] = 1
    else:
        data, col_ind, row_ind = compute_similarity_matrix_sparse_nogil(mzvec, data_p, data_sizes, mz_tolerance, min_matched_peaks, callback)
        matrix = csr_matrix((arr_from_1d_vector(data, np.dtype(np.float32)),
                             (arr_from_1d_vector(col_ind, np.dtype(np.int32)),
                             arr_from_1d_vector(row_ind, np.dtype(np.int32)))),
                            dtype=np.float32)

    data_sizes.clear()
    data_sizes.shrink_to_fit()
    return matrix
