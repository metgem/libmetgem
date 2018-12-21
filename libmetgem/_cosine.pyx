# cython: language_level=3
# distutils: language=c++

cimport cython
from cython.parallel import prange
from cython.view cimport array as cvarray
import numpy as np
cimport numpy as np
from libcpp cimport bool
from libcpp.algorithm cimport sort
from libcpp.vector cimport vector
from libc.math cimport fabs

from ._common cimport peak_t, np_arr_pointer

DEF MZ = 0
DEF INTENSITY = 1

cdef extern from "<algorithm>" namespace "std" nogil:
    void fill[Iter, T](Iter first, Iter last, T value)
    
cdef packed struct score_t:
  int ix1, ix2
  double value
  
cdef bool compareByScore(const score_t &a, const score_t &b) nogil:
    return a.value > b.value
  
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double cosine_score_nogil(double spectrum1_mz, peak_t *spectrum1_data, int spectrum1_size,
                              double spectrum2_mz, peak_t *spectrum2_data, int spectrum2_size,
                              double mz_tolerance, int min_matched_peaks) nogil:
    cdef double dm
    cdef vector[score_t] scores
    cdef score_t pscore
    cdef int i, j
    cdef vector[bool] peak_used1
    cdef vector[bool] peak_used2
    cdef double score = 0.
    cdef int num_matched_peaks = 0
    cdef int ix1, ix2
    
    if spectrum1_size == 0 or spectrum2_size == 0:
        return 0.
    
    dm = spectrum1_mz - spectrum2_mz

    if dm == 0.:
        for i in range(spectrum1_size):
            for j in range(spectrum2_size):
                if fabs(spectrum2_data[j].mz - spectrum1_data[i].mz) <= mz_tolerance:
                    pscore.ix1 = i
                    pscore.ix2 = j
                    pscore.value = spectrum1_data[i].intensity * spectrum2_data[j].intensity
                    scores.push_back(pscore)

    else:
        for i in range(spectrum1_size):
            for j in range(spectrum2_size):
                if fabs(spectrum2_data[j].mz - spectrum1_data[i].mz) <= mz_tolerance or fabs(spectrum2_data[j].mz - spectrum1_data[i].mz + dm) <= mz_tolerance:
                    pscore.ix1 = i
                    pscore.ix2 = j
                    pscore.value = spectrum1_data[i].intensity * spectrum2_data[j].intensity
                    scores.push_back(pscore)
    
    if scores.size() == 0:
        return 0.
    
    sort(scores.begin(), scores.end(), &compareByScore)
    
    peak_used1.resize(spectrum1_size)
    fill(peak_used1.begin(), peak_used1.end(), 0)
    peak_used2.resize(spectrum2_size)
    fill(peak_used2.begin(), peak_used2.end(), 0)
    
    for i in range(scores.size()):
        ix1 = scores[i].ix1
        ix2 = scores[i].ix2
        if not peak_used1[ix1] and not peak_used2[ix2]:
            score += scores[i].value
            peak_used1[ix1] = 1
            peak_used2[ix2] = 1
            num_matched_peaks += 1
            
    if num_matched_peaks < min_matched_peaks:
        return 0.

    return score

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float[:,:] compute_distance_matrix_nogil(vector[double] mzvec, vector[peak_t *] datavec, vector[int] data_sizes, double mz_tolerance, int min_matched_peaks, object callback=None):
    cdef:
        int i, j
        int size = mzvec.size()
        float[:,:] matrix = cvarray(shape=(size, size), itemsize=sizeof(float), format='f')
        bool has_callback = callback is not None
    
    with nogil:
        for i in prange(size, schedule='guided'):
            matrix[i, i] = 1
            for j in range(i):
                matrix[i, j] = matrix[j, i] = cosine_score_nogil(mzvec[i], datavec[i], data_sizes[i],
                                                               mzvec[j], datavec[j], data_sizes[j],
                                                               mz_tolerance, min_matched_peaks)
            if has_callback:
                with gil:
                    if not callback(i):
                        return matrix
    
    return matrix
        
def cosine_score(double spectrum1_mz, np.ndarray[np.float32_t, ndim=2] spectrum1_data, double spectrum2_mz, np.ndarray[np.float32_t, ndim=2] spectrum2_data, double mz_tolerance, int min_matched_peaks):
    return cosine_score_nogil(spectrum1_mz, np_arr_pointer(spectrum1_data), spectrum1_data.shape[0],
                              spectrum2_mz, np_arr_pointer(spectrum2_data), spectrum2_data.shape[0],
                              mz_tolerance, min_matched_peaks)
    
def compute_distance_matrix(vector[double] mzvec, list datavec, double mz_tolerance, int min_matched_peaks, object callback=None):
    cdef:
        np.ndarray[np.float32_t, ndim=2] tmp_array
        int size = mzvec.size()
        int i
        vector[peak_t *] data_p
        vector[int] data_sizes
        
    data_p.resize(size)
    data_sizes.resize(size)
    for i, tmp_array in enumerate(datavec):
        data_p[i] = np_arr_pointer(tmp_array)
        data_sizes[i] = tmp_array.shape[0]
        
    matrix = np.asarray(compute_distance_matrix_nogil(mzvec, data_p, data_sizes, mz_tolerance, min_matched_peaks, callback))
    matrix[matrix>1] = 1
    return matrix
