# distutils: language=c++

cimport cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython.view cimport array as cvarray
from libcpp.algorithm cimport sort
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.utility cimport pair
from libc.math cimport fabs

cdef extern from "<algorithm>" namespace "std" nogil:
    void fill[Iter, T](Iter first, Iter last, T value)
    
cdef packed struct score_t:
  int ix1, ix2
  float value

ctypedef pair[float, float] peak_t
  
cdef bool compareByScore(const score_t &a, const score_t &b) nogil:
    return a.value > b.value
    
cdef bool compareByIntensity(const peak_t &a, const peak_t &b) nogil:
    return a.second > b.second
    
DEF MZ = 0
DEF INTENSITY = 1
  
@cython.boundscheck(False)
@cython.wraparound(False)
cdef float cosine_score_nogil(float spectrum1_mz, float[:,:] spectrum1_data, float spectrum2_mz, float[:,:] spectrum2_data, float mz_tolerance, int min_matched_peaks) nogil:
    cdef float dm
    cdef vector[score_t] scores
    cdef score_t pscore
    cdef int i, j
    cdef vector[bool] peak_used1
    cdef vector[bool] peak_used2
    cdef float score = 0.
    cdef int num_matched_peaks = 0
    cdef int ix1, ix2
    cdef Py_ssize_t size1 = spectrum1_data.shape[0]
    cdef Py_ssize_t size2 = spectrum2_data.shape[0]
    
    dm = spectrum1_mz - spectrum2_mz

    if dm == 0.:
        for i in range(size1):
            for j in range(size2):
                if fabs(spectrum2_data[j, MZ] - spectrum1_data[i, MZ]) <= mz_tolerance:
                    pscore.ix1 = i
                    pscore.ix2 = j
                    pscore.value = spectrum1_data[i, INTENSITY] * spectrum2_data[j, INTENSITY]
                    scores.push_back(pscore)

    else:
        for i in range(size1):
            for j in range(size2):
                if fabs(spectrum2_data[j, MZ] - spectrum1_data[i, MZ]) <= mz_tolerance or fabs(spectrum2_data[j, MZ] - spectrum1_data[i, MZ] + dm) <= mz_tolerance:
                    pscore.ix1 = i
                    pscore.ix2 = j
                    pscore.value = spectrum1_data[i, INTENSITY] * spectrum2_data[j, INTENSITY]
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
cdef float[:,:] compute_distance_matrix_nogil(vector[float] mzvec, vector[float[:,:]] datavec, float mz_tolerance, int min_matched_peaks, object callback=None):
    cdef int i, j
    cdef int size = mzvec.size()
    cdef float[:,:] matrix = cvarray(shape=(size, size), itemsize=sizeof(float), format='f')
    cdef bool has_callback = callback is not None
    
    with nogil:
        for i in prange(size, schedule='guided'):
            for j in range(i):
                matrix[i,j] = matrix[j,i] = cosine_score_nogil(mzvec[i], datavec[i], mzvec[j], datavec[j], mz_tolerance, min_matched_peaks)
            if has_callback:
                with gil:
                    callback(size-1)
                    
    with nogil:
        for i in prange(size):
            matrix[i,i] = 1
    if has_callback:
        callback(size)
    
    return matrix
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef vector[peak_t] filter_data_nogil(float[:,:] data, float mz_parent, int min_intensity, int parent_filter_tolerance, int matched_peaks_window, int min_matched_peaks_search) nogil:
    cdef Py_ssize_t size = data.shape[0]
    cdef int i, j, count=0
    cdef float mz, intensity, max_intensity=0
    cdef float abs_min_intensity
    cdef vector[peak_t] peaks
    cdef vector[peak_t] peaks2
    cdef peak_t peak
    
    # Sort data array by decreasing intensities
    sort(<peak_t*>&data[0,0], (<peak_t*>&data[0,0]) + size, compareByIntensity)

    # Filter out peaks with mz below 50 Da or with mz in `mz_parent` +- `parent_filter_tolerance` or with intensity < `min_intensity` % of maximum intensity
    # Maximum intensity is calculated from peaks not filtered out by mz filters
    peaks.reserve(size)
    for i in range(size):
        mz = data[i, MZ]
        if 50 <= mz <= mz_parent - parent_filter_tolerance or mz >= mz_parent + parent_filter_tolerance:  # mz filter
            intensity = data[i, INTENSITY]
            if intensity < abs_min_intensity:  # intensity filter
                break
            elif intensity > max_intensity:
                max_intensity = intensity
                abs_min_intensity = min_intensity * max_intensity / 100
            
            peak.first = mz
            peak.second = intensity
            peaks.push_back(peak)
    
    # Window rank filter: For each peak, keep it only if it is in the top `min_matched_peaks_search` peaks in the +/- `matched_peaks_window` range
    size = peaks.size()
    peaks2.reserve(size)
    for i in range(size):
        mz = peaks[i].first
        count = 0
        for j in range(size):
            if mz - matched_peaks_window <= peaks[j].first <= mz + matched_peaks_window:
                if j == i:
                    peaks2.push_back(peaks[i])
                    break
                count += 1
                if count >= min_matched_peaks_search:
                    break
        
    return peaks2 #<float[:peaks.size(),:2]>(<float*>peaks.data())
        
def cosine_score(float spectrum1_mz, float[:,:] spectrum1_data, float spectrum2_mz, float[:,:] spectrum2_data, float mz_tolerance, int min_matched_peaks):
    return cosine_score_nogil(spectrum1_mz, spectrum1_data, spectrum2_mz, spectrum2_data, mz_tolerance, min_matched_peaks)
    
def compute_distance_matrix(vector[float] mzvec, vector[float[:,:]] datavec, float mz_tolerance, int min_matched_peaks, object callback=None):
    matrix = np.asarray(compute_distance_matrix_nogil(mzvec, datavec, mz_tolerance, min_matched_peaks, callback))
    matrix[matrix>1] = 1
    return matrix
    
def filter_data(np.ndarray[np.float32_t, ndim=2] data, float mz_parent, int min_intensity, int parent_filter_tolerance, int matched_peaks_window, int min_matched_peaks_search):
    return np.array(filter_data_nogil(data, mz_parent, min_intensity, parent_filter_tolerance, matched_peaks_window, min_matched_peaks_search), dtype=np.float32)
    