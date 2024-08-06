# cython: language_level=3
# cython: linetrace=True
# distutils: language=c++

cimport cython
from cython.parallel import prange
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.algorithm cimport sort

cdef extern from "<algorithm>" namespace "std" nogil:
    Iter min_element[Iter, Compare](Iter first, Iter last, Compare comp) except +

from ._common cimport (peak_t, arr_from_peaks_vector, np_arr_pointer,
                       norm_method_t, str_to_norm_method)

cdef bool compareByIntensity(const peak_t &a, const peak_t &b) nogil:
    return a.intensity > b.intensity

cdef bool compareByMz(const peak_t &a, const peak_t &b) nogil:
    return a.mz < b.mz
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef vector[peak_t] parent_filter_nogil(double mz_parent, vector[peak_t] data,
                                        int min_intensity,
                                        int parent_filter_tolerance,
                                        double mz_min = 50.) noexcept nogil:
    cdef:
        int i
        float mz, intensity
        double abs_min_intensity = -1.
        vector[peak_t] result
        peak_t peak
        size_t size
            
    # Filter out peaks with mz below `mz_min` Da or with mz in `mz_parent` +- `parent_filter_tolerance` or with intensity < `min_intensity` % of maximum intensity
    # Maximum intensity is calculated from peaks not filtered out by mz filters
    size = data.size()
    result.reserve(size)
    for i in range(size):
        mz = data[i].mz
        if mz_min <= mz <= mz_parent - parent_filter_tolerance or mz >= mz_parent + parent_filter_tolerance:  # mz filter
            intensity = data[i].intensity
            
            if abs_min_intensity < 0:
                abs_min_intensity = min_intensity / 100. * intensity
            
            if intensity < abs_min_intensity:  # intensity filter
                break
            
            peak.mz = mz
            peak.intensity = intensity
            result.push_back(peak)
    
    return result
   

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)    
cdef vector[peak_t] window_rank_filter_nogil(vector[peak_t] data,
                                             int matched_peaks_window,
                                             int min_matched_peaks_search) noexcept nogil:
    cdef:
        int i, j, count=0
        float mz
        vector[peak_t] result
        peak_t peak
        size_t size
    
    # Window rank filter: For each peak, keep it only if it is in the top `min_matched_peaks_search` peaks in the +/- `matched_peaks_window` range
    size = data.size()
    result.reserve(size)
    for i in range(size):
        peak = data[i]
        mz = peak.mz
        count = 0
        for j in range(size):
            if matched_peaks_window==0 or mz - matched_peaks_window <= data[j].mz <= mz + matched_peaks_window:
                if j == i:
                    result.push_back(peak)
                    break
                count += 1
                if count >= min_matched_peaks_search > 0:
                    break
        
    return result
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef vector[peak_t] square_root_data_nogil(vector[peak_t] data) noexcept nogil:
    cdef:
        int i
        size_t size
    
    size = data.size()
    for i in range(size):
        data[i].intensity = <float> (sqrt(data[i].intensity) * 10)  # Use square root of intensities to minimize/maximize effects of high/low intensity peaks
        
    return data
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef vector[peak_t] normalize_data_nogil(vector[peak_t] data, norm_method_t norm_method=norm_method_t.dot) noexcept nogil:
    cdef:
        int i
        double accum = 0.
        size_t size = data.size()
    
    if norm_method == norm_method_t.sum:            
        # Normalize the intensity to sum to 1
        for i in range(size):
            accum += data[i].intensity
            
        for i in range(size):
            data[i].intensity /= accum
    else:
        # Normalize data to norm 1
        for i in range(size):
            accum += data[i].intensity * data[i].intensity  # Calculate dot product for later normalization
            
        accum = sqrt(accum)
        for i in range(size):
            data[i].intensity = <float> (data[i].intensity / accum)
            
    return data
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef vector[peak_t] filter_data_nogil(double mz_parent, const peak_t *data,
                                      np.npy_intp data_size, int min_intensity,
                                      int parent_filter_tolerance,
                                      int matched_peaks_window,
                                      int min_matched_peaks_search,
                                      double mz_min,
                                      bool square_root=True,
                                      norm_method_t norm_method=norm_method_t.dot) noexcept nogil:
    cdef vector[peak_t] peaks

    if data_size == 0:
        return peaks

    peaks.assign(data, data+data_size)
        
    # Sort data array by decreasing intensities
    sort(peaks.begin(), peaks.end(), &compareByIntensity)
    
    if min_intensity > 0 or parent_filter_tolerance > 0 or (<peak_t> (min_element(peaks.begin(), peaks.end(), &compareByMz)[0])).mz < mz_min:
        peaks = parent_filter_nogil(mz_parent, peaks, min_intensity, parent_filter_tolerance, mz_min)
        
    if matched_peaks_window > 0 and min_matched_peaks_search > 0:
        peaks = window_rank_filter_nogil(peaks, matched_peaks_window, min_matched_peaks_search)
        
    if square_root:
        peaks = square_root_data_nogil(peaks)
    
    peaks = normalize_data_nogil(peaks, norm_method)
    
    return peaks
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)    
cdef vector[vector[peak_t]] filter_data_multi_nogil(vector[double] mzvec,
                                                    vector[peak_t *] datavec,
                                                    vector[np.npy_intp] data_sizes,
                                                    int min_intensity,
                                                    int parent_filter_tolerance,
                                                    int matched_peaks_window,
                                                    int min_matched_peaks_search,
                                                    double mz_min,
                                                    bool square_root=True,
                                                    norm_method_t norm_method=norm_method_t.dot,
                                                    object callback=None) noexcept nogil:
    cdef:
        vector[vector[peak_t]] spectra
        int i
        size_t size = mzvec.size()
        np.npy_intp data_size
        peak_t* data_p = NULL
        bool has_callback = callback is not None
    
    spectra.resize(size)
    for i in prange(<int>size, schedule='guided'):
        data_size = data_sizes[i]
        data_p = datavec[i]
        spectra[i] = filter_data_nogil(mzvec[i], data_p, data_size,
                                       min_intensity, parent_filter_tolerance,
                                       matched_peaks_window, min_matched_peaks_search,
                                       mz_min, square_root, norm_method)
        if has_callback and i % 100 == 0:
            with gil:
                callback(100)
        
    if has_callback and size//100 != 0:
        with gil:
            callback(size//100)
        
    return spectra
      

def square_root_data(np.ndarray[np.float32_t, ndim=2] data):
    cdef np.ndarray[np.float32_t, ndim=2] squared
    cdef vector[peak_t] peaks
    cdef peak_t* data_p = <peak_t*>np_arr_pointer(data)

    if data.shape[0] == 0:
        return peaks

    peaks.assign(data_p, data_p+data.shape[0])
    peaks = square_root_data_nogil(peaks)
    
    if peaks.size() == 0:
        return np.empty((0,2), dtype=np.float32)
        
    squared = arr_from_peaks_vector(peaks)
    
    # Free memory
    peaks.clear()
    peaks.shrink_to_fit()
    
    return squared
    
    
def normalize_data(np.ndarray[np.float32_t, ndim=2] data, norm='dot'):
    cdef:
        np.ndarray[np.float32_t, ndim=2] normalized
        vector[peak_t] peaks
        peak_t* data_p = <peak_t*>np_arr_pointer(data)
        norm_method_t norm_method = str_to_norm_method(norm)

    if data.shape[0] == 0:
        return peaks

    peaks.assign(data_p, data_p+data.shape[0])
    peaks = normalize_data_nogil(peaks, norm_method)
    
    if peaks.size() == 0:
        return np.empty((0,2), dtype=np.float32)
        
    normalized = arr_from_peaks_vector(peaks)
    
    # Free memory
    peaks.clear()
    peaks.shrink_to_fit()
    
    return normalized


def filter_data(double mz_parent, np.ndarray[np.float32_t, ndim=2] data,
                int min_intensity, int parent_filter_tolerance,
                int matched_peaks_window, int min_matched_peaks_search,
                double mz_min = 50., bool square_root=True, str norm='dot'):
    cdef:
        np.ndarray[np.float32_t, ndim=2] filtered
        norm_method_t norm_method = str_to_norm_method(norm)
        vector[peak_t] peaks

    peaks = filter_data_nogil(mz_parent, <peak_t*>np_arr_pointer(data), data.shape[0],
                              min_intensity, parent_filter_tolerance,
                              matched_peaks_window, min_matched_peaks_search,
                              mz_min, square_root, norm_method)
    
    if peaks.size() == 0:
        return np.empty((0,2), dtype=np.float32)
        
    filtered = arr_from_peaks_vector(peaks)
    
    # Free memory
    peaks.clear()
    peaks.shrink_to_fit()
    
    return filtered

    
def filter_data_multi(vector[double] mzvec, list datavec, int min_intensity,
                      int parent_filter_tolerance, int matched_peaks_window,
                      int min_matched_peaks_search, double mz_min = 50.,
                      bool square_root=True, str norm='dot', object callback=None):
    cdef:
        list filtered = []
        norm_method_t norm_method = str_to_norm_method(norm)
        np.ndarray[np.float32_t, ndim=2] tmp_array
        vector[vector[peak_t]] spectra
        vector[peak_t] peaks
        size_t size = mzvec.size()
        int i
        vector[peak_t *] data_p
        vector[np.npy_intp] data_sizes
        
    data_p.resize(size)
    data_sizes.resize(size)
    for i, tmp_array in enumerate(datavec):
        data_p[i] = <peak_t*>np_arr_pointer(tmp_array)
        data_sizes[i] = tmp_array.shape[0]
    
    spectra = filter_data_multi_nogil(mzvec, data_p, data_sizes,
                                      min_intensity, parent_filter_tolerance,
                                      matched_peaks_window, min_matched_peaks_search,
                                      mz_min, square_root, norm_method, callback)
    
    for i in range(size):
        peaks = spectra[i]
        if peaks.size() == 0:
            tmp_array = np.empty((0,2), dtype=np.float32)
        else:
            tmp_array = arr_from_peaks_vector(peaks)
            
        filtered.append(tmp_array)
        
    # Free memory
    spectra.clear()
    spectra.shrink_to_fit()
    peaks.clear()
    peaks.shrink_to_fit()
    data_sizes.clear()
    data_sizes.shrink_to_fit()
    
    return filtered
