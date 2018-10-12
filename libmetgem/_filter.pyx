# cython: language_level=3
# distutils: language=c++

cimport cython
from cython.parallel import prange
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.algorithm cimport sort

from ._common cimport peak_t, arr_from_vector, np_arr_pointer

DEF MZ = 0
DEF INTENSITY = 1

cdef bool compareByIntensity(const peak_t &a, const peak_t &b) nogil:
    return a.intensity > b.intensity
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef vector[peak_t] filter_data_nogil(double mz_parent, const peak_t *data, int data_size, int min_intensity, int parent_filter_tolerance, int matched_peaks_window, int min_matched_peaks_search) nogil:
    cdef int i, j, count=0
    cdef double mz, intensity
    cdef double abs_min_intensity = -1.
    cdef vector[peak_t] peaks
    cdef vector[peak_t] peaks2
    cdef vector[peak_t] peaks3
    cdef peak_t peak
    cdef int size
    cdef double dot_product

    peaks.assign(data, data+data_size)
    
    # Sort data array by decreasing intensities
    sort(peaks.begin(), peaks.end(), compareByIntensity)
            
    # Filter out peaks with mz below 50 Da or with mz in `mz_parent` +- `parent_filter_tolerance` or with intensity < `min_intensity` % of maximum intensity
    # Maximum intensity is calculated from peaks not filtered out by mz filters
    size = peaks.size()
    peaks2.reserve(size)
    for i in range(size):
        mz = peaks[i].mz
        if 50 <= mz <= mz_parent - parent_filter_tolerance or mz >= mz_parent + parent_filter_tolerance:  # mz filter
            intensity = peaks[i].intensity
            
            if abs_min_intensity < 0:
                abs_min_intensity = min_intensity / 100. * intensity
            
            if intensity < abs_min_intensity:  # intensity filter
                break
            
            peak.mz = mz
            peak.intensity = intensity
            peaks2.push_back(peak)
    
    # Window rank filter: For each peak, keep it only if it is in the top `min_matched_peaks_search` peaks in the +/- `matched_peaks_window` range
    dot_product = 0.
    size = peaks2.size()
    peaks3.reserve(size)
    for i in range(size):
        peak = peaks2[i]
        mz = peak.mz
        count = 0
        for j in range(size):
            if matched_peaks_window==0 or mz - matched_peaks_window <= peaks2[j].mz <= mz + matched_peaks_window:
                if j == i:
                    peak.intensity = sqrt(peak.intensity) * 10  # Use square root of intensities to minimize/maximize effects of high/low intensity peaks
                    peaks3.push_back(peak)
                    dot_product += peak.intensity * peak.intensity # Calculate dot product for later normalization
                    break
                count += 1
                if count >= min_matched_peaks_search > 0:
                    break
                    
    # Normalize data to norm 1
    dot_product = sqrt(dot_product)
    for i in range(peaks3.size()):
        peaks3[i].intensity = peaks3[i].intensity / dot_product
        
    return peaks3
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)    
cdef vector[vector[peak_t]] filter_data_multi_nogil(vector[double] mzvec, vector[peak_t *] datavec, vector[int] data_sizes, int min_intensity,
                                                    int parent_filter_tolerance, int matched_peaks_window, int min_matched_peaks_search, object callback=None) nogil:
    cdef vector[vector[peak_t]] spectra
    cdef int i
    cdef int size = mzvec.size()
    cdef int data_size
    cdef peak_t* data_p = NULL
    cdef bool has_callback = callback is not None
    
    spectra.resize(size)
    for i in prange(size, schedule='guided'):
        data_size = data_sizes[i]
        data_p = datavec[i]
        spectra[i] = filter_data_nogil(mzvec[i], data_p, data_size, min_intensity, parent_filter_tolerance, matched_peaks_window, min_matched_peaks_search)
        if has_callback and i % 100 == 0:
            with gil:
                callback(100)
        
    if has_callback and size//100 != 0:
        with gil:
            callback(size//100)
        
    return spectra
      
def filter_data(double mz_parent, np.ndarray[np.float32_t, ndim=2] data, int min_intensity, int parent_filter_tolerance, int matched_peaks_window, int min_matched_peaks_search):
    cdef np.ndarray[np.float32_t, ndim=2] filtered
    cdef vector[peak_t] peaks = filter_data_nogil(mz_parent, np_arr_pointer(data), data.shape[0], min_intensity, parent_filter_tolerance, matched_peaks_window, min_matched_peaks_search)
    
    if peaks.size() == 0:
        return np.empty((0,2), dtype=np.float32)
        
    filtered = np.asarray(arr_from_vector(peaks))
    
    return filtered
    
def filter_data_multi(vector[double] mzvec, list datavec, int min_intensity, int parent_filter_tolerance,
                      int matched_peaks_window, int min_matched_peaks_search, object callback=None):
    cdef:
        list filtered = []
        np.ndarray[np.float32_t, ndim=2] tmp_array
        vector[vector[peak_t]] spectra
        vector[peak_t] peaks
        int size = mzvec.size()
        int i
        vector[peak_t *] data_p
        vector[int] data_sizes
        
    data_p.resize(size)
    data_sizes.resize(size)
    for i, tmp_array in enumerate(datavec):
        data_p[i] = np_arr_pointer(tmp_array)
        data_sizes[i] = tmp_array.shape[0]
    
    spectra = filter_data_multi_nogil(mzvec, data_p, data_sizes, min_intensity, parent_filter_tolerance, matched_peaks_window, min_matched_peaks_search, callback)
    
    for i in range(size):
        peaks = spectra[i]
        if peaks.size() == 0:
            tmp_array = np.empty((0,2), dtype=np.float32)
        else:
            tmp_array = np.asarray(arr_from_vector(peaks))
            
        filtered.append(tmp_array)
    
    return filtered