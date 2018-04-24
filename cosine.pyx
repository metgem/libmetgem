# distutils: language=c++
import cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libcpp.algorithm cimport sort
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.math cimport fabs

cdef extern from "<algorithm>" namespace "std" nogil:
    void fill[Iter, T](Iter first, Iter last, T value)
    
cdef packed struct score_t:
  int ix1, ix2
  double value

cdef bool compareByScore(const score_t &a, const score_t &b) nogil:
    return a.value > b.value
    
ctypedef void (*callback_t)(int i)
  
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef float cosine_score_nogil(double spectrum1_mz, float[:,:] spectrum1_data, double spectrum2_mz, float[:,:] spectrum2_data, float mz_tolerance, int min_matched_peaks) nogil:
    cdef double dm
    cdef vector[score_t] scores
    cdef score_t pscore
    cdef int i, j
    cdef vector[bool] peak_used1
    cdef vector[bool] peak_used2
    cdef float score = 0.
    cdef int num_matched_peaks = 0
    cdef int ix1, ix2
    
    dm = spectrum1_mz - spectrum2_mz

    if dm == 0.:
        for i in range(spectrum1_data.shape[0]):
            for j in range(spectrum2_data.shape[0]):
                if fabs(spectrum2_data[j,0] - spectrum1_data[i,0]) <= mz_tolerance:
                    pscore.ix1 = i
                    pscore.ix2 = j
                    pscore.value = spectrum1_data[i, 1] * spectrum2_data[j, 1]
                    scores.push_back(pscore)

    else:
        for i in range(spectrum1_data.shape[0]):
            for j in range(spectrum2_data.shape[0]):
                if fabs(spectrum2_data[j,0] - spectrum1_data[i,0]) <= mz_tolerance or fabs(spectrum2_data[j,0] - spectrum1_data[i,0] + dm) <= mz_tolerance:
                    pscore.ix1 = i
                    pscore.ix2 = j
                    pscore.value = spectrum1_data[i, 1] * spectrum2_data[j, 1]
                    scores.push_back(pscore)
    
    if scores.size() == 0:
        return 0.
    
    sort(scores.begin(), scores.end(), compareByScore)
    
    peak_used1.resize(spectrum1_data.shape[0])
    fill(peak_used1.begin(), peak_used1.end(), 0)
    peak_used2.resize(spectrum2_data.shape[0])
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
@cython.initializedcheck(False)
cdef float[:,:] compute_distance_matrix_nogil(vector[double] mzvec, vector[float[:,:]] datavec, float mz_tolerance, int min_matched_peaks, object callback=None):
    cdef int i, j
    cdef int size = mzvec.size()
    cdef float[:,:] matrix = np.empty((size, size), dtype=np.float32)
    cdef bool has_callback = callback is not None
    
    with nogil:
        for i in prange(size, schedule='guided'):
            for j in range(i):
                matrix[i,j] = matrix[j,i] = cosine_score_nogil(mzvec[i], datavec[i], mzvec[j], datavec[j], mz_tolerance, min_matched_peaks)
            if has_callback:
                with gil:
                    callback(size-1)
        for i in prange(size):
            matrix[i,i] = 1
    if has_callback:
        callback(size)
    
    return matrix
        
def cosine_score(double spectrum1_mz, float[:,:] spectrum1_data, double spectrum2_mz, float[:,:] spectrum2_data, float mz_tolerance, int min_matched_peaks):
    return cosine_score_nogil(spectrum1_mz, spectrum1_data, spectrum2_mz, spectrum2_data, mz_tolerance, min_matched_peaks)
    
def compute_distance_matrix(vector[double] mzvec, vector[float[:,:]] datavec, float mz_tolerance, int min_matched_peaks, object callback=None):
    matrix = np.asarray(compute_distance_matrix_nogil(mzvec, datavec, mz_tolerance, min_matched_peaks, callback))
    matrix[matrix>1] = 1
    return matrix
    