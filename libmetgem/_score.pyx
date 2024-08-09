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
from libcpp.numeric cimport accumulate
from libc.math cimport fabs, log2f, logf, powf
from libc.stdint cimport uint16_t, uint8_t
from scipy.sparse import csr_matrix

from ._common cimport (score_t, np_arr_pointer,
                       arr_from_peaks_vector,
                       arr_from_score_vector,
                       arr_from_1d_vector,
                       score_algorithm_t,
                       str_to_score_algorithm,
                       norm_method_t)
from ._filter cimport normalize_data_nogil
                       
cdef double addScore(double sum, const score_t &b) noexcept nogil:
    return sum + b.value
  
cdef bool compareByScore(const score_t &a, const score_t &b) noexcept nogil:
    return a.value > b.value


@cython.cdivision(True)
cdef double partial_score_nogil(float intensity1, float intensity2, score_algorithm_t score_algorithm=score_algorithm_t.cosine) noexcept nogil:
    cdef:
        float intensity_sum
        
    if score_algorithm == score_algorithm_t.entropy or score_algorithm == score_algorithm_t.weighted_entropy:
        intensity_sum = intensity1 + intensity2
        score = intensity_sum * log2f(intensity_sum) - intensity1 * log2f(intensity1) - intensity2 * log2f(intensity2)
    else:
        score = intensity1 * intensity2
        
    return score


@cython.cdivision(True)
cdef double generic_score_nogil(double spectrum1_mz,
                  const peak_t *spectrum1_data,
                  np.npy_intp spectrum1_size,
                  double spectrum2_mz,
                  const peak_t *spectrum2_data,
                  np.npy_intp spectrum2_size,
                  double mz_tolerance,
                  int min_matched_peaks,
                  score_algorithm_t score_algorithm=score_algorithm_t.cosine
                  ) noexcept nogil:
    cdef:
        double result
        vector[score_t] matches = compare_spectra_nogil(
            spectrum1_mz,
            spectrum1_data,
            spectrum1_size,
            spectrum2_mz,
            spectrum2_data,
            spectrum2_size,
            mz_tolerance,
            score_algorithm=score_algorithm)
                
    if matches.size() < min_matched_peaks:
        return 0.
    
    result = accumulate(matches.begin(), matches.end(), 0., &addScore)
    matches.clear()
    matches.shrink_to_fit()
    if score_algorithm in (score_algorithm_t.entropy, score_algorithm_t.weighted_entropy):
        result /= 2
    return result
    

@cython.cdivision(True)
cdef vector[score_t] compare_spectra_nogil(double spectrum1_mz,
                                           const peak_t *spectrum1_data,
                                           np.npy_intp spectrum1_size,
                                           double spectrum2_mz,
                                           const peak_t *spectrum2_data,
                                           np.npy_intp spectrum2_size,
                                           double mz_tolerance,
                                           score_algorithm_t score_algorithm=score_algorithm_t.cosine) noexcept nogil:
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
        vector[peak_t] data1_vec
        vector[peak_t] data2_vec
        const peak_t *data1
        const peak_t *data2
    
    if spectrum1_size == 0 or spectrum2_size == 0:
        return matches
    
    dm = spectrum1_mz - spectrum2_mz
    
    if score_algorithm == score_algorithm_t.weighted_entropy:
        # Apply the weights to the peaks.
        data1_vec = apply_weight_to_intensity_no_gil(spectrum1_data, spectrum1_size)
        data2_vec = apply_weight_to_intensity_no_gil(spectrum2_data, spectrum2_size)
        data1 = <peak_t*>&data1_vec[0]
        data2 = <peak_t*>&data2_vec[0]
    else:
        data1 = spectrum1_data
        data2 = spectrum2_data

    scores.reserve(spectrum1_size * spectrum2_size)
    if dm == 0.:
        for i in range(spectrum1_size):
            for j in range(spectrum2_size):
                diff = data2[j].mz - data1[i].mz
                if fabs(diff) <= mz_tolerance:
                    pscore.ix1 = i
                    pscore.ix2 = j
                    pscore.value = partial_score_nogil(data1[i].intensity, data2[j].intensity, score_algorithm)
                    pscore.type = SpectraMatchState.fragment
                    scores.push_back(pscore)
    else:
        for i in range(spectrum1_size):
            for j in range(spectrum2_size):
                diff = data2[j].mz - data1[i].mz
                if fabs(diff) <= mz_tolerance:
                    pscore.ix1 = i
                    pscore.ix2 = j
                    pscore.value = partial_score_nogil(data1[i].intensity, data2[j].intensity, score_algorithm)
                    pscore.type = SpectraMatchState.fragment
                    scores.push_back(pscore)
                elif fabs(diff + dm) <= mz_tolerance:
                    pscore.ix1 = i
                    pscore.ix2 = j
                    pscore.value = partial_score_nogil(data1[i].intensity, data2[j].intensity, score_algorithm)
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


# Calculate spectral entropy of a peaks. The peaks intensity need to be prenormalized.
@cython.cdivision(True)
cdef double spectral_entropy_no_gil(const peak_t* spectrum_data, np.npy_intp spectrum_size) noexcept nogil:
    cdef:
        float intensity_sum = 0.
        float intensity = 0.
        double entropy = 0.
        
    if spectrum_size == 0:
        return 0.

    for i in range(spectrum_size):
        intensity_sum += spectrum_data[i].intensity

    if intensity_sum == 0:
        return 0.
        
    for i in range(spectrum_size):
        intensity = spectrum_data[i].intensity / intensity_sum
        entropy -= intensity * logf(intensity)

    return entropy


# Apply weight to a peaks by spectral entropy.
# The peaks intensity need to be prenormalized.
@cython.cdivision(True)
cdef vector[peak_t] apply_weight_to_intensity_no_gil(const peak_t *spectrum_data, np.npy_intp spectrum_size) noexcept nogil:
    cdef:
        vector[peak_t] weighted
        double entropy = spectral_entropy_no_gil(spectrum_data, spectrum_size)
        double weight = 0.25 + 0.25 * entropy
        int i
    
    weighted.reserve(spectrum_size)
    weighted.assign(spectrum_data, spectrum_data+spectrum_size)
    
    if entropy < 3:        
        # Calculate the sum of intensity.
        for i in range(spectrum_size):
            weighted[i].intensity = powf(weighted[i].intensity, weight)

        # Normalize the intensity.
        weighted = normalize_data_nogil(weighted, norm_method_t.sum)
    
    return weighted


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float[:,:] compute_similarity_matrix_dense_nogil(vector[double] mzvec,
                                                      vector[peak_t *] datavec,
                                                      vector[np.npy_intp] data_sizes,
                                                      double mz_tolerance,
                                                      int min_matched_peaks,
                                                      score_algorithm_t score_algorithm=score_algorithm_t.cosine,
                                                      object callback=None):
    cdef:
        int i, j
        size_t size = datavec.size()
        float[:,:] matrix = cvarray(shape=(size, size), itemsize=sizeof(float), format='f')
        bool has_callback = callback is not None
    
    with nogil:
        for i in prange(<int>size, schedule='guided'):
            matrix[i, i] = 1
            for j in range(i):
                matrix[i, j] = matrix[j, i] = <float> generic_score_nogil(
                    mzvec[i], datavec[i], data_sizes[i],
                    mzvec[j], datavec[j], data_sizes[j],
                    mz_tolerance, min_matched_peaks,
                    score_algorithm)
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
           score_algorithm_t score_algorithm=score_algorithm_t.cosine,
           object callback=None):
    cdef:
        int i, j
        size_t size = datavec.size()
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
                score = <float> generic_score_nogil(
                    mzvec[i], datavec[i], data_sizes[i],
                    mzvec[j], datavec[j], data_sizes[j],
                    mz_tolerance, min_matched_peaks,
                    score_algorithm)
                
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


def compare_spectra(double spectrum1_mz,
                    np.ndarray[np.float32_t, ndim=2] spectrum1_data,
                    double spectrum2_mz,
                    np.ndarray[np.float32_t, ndim=2] spectrum2_data,
                    double mz_tolerance,
                    str scoring = 'cosine'):
    cdef:
        vector[score_t] matches
        score_algorithm_t score_algorithm

    if scoring == 'weighted_entropy':
        score_algorithm = score_algorithm_t.weighted_entropy
    elif scoring == 'entropy':
        score_algorithm = score_algorithm_t.entropy
    else:
        score_algorithm = score_algorithm_t.cosine
    
    matches = compare_spectra_nogil(
        spectrum1_mz,
        <peak_t*>np_arr_pointer(spectrum1_data),
        spectrum1_data.shape[0],
        spectrum2_mz,
        <peak_t*>np_arr_pointer(spectrum2_data),
        spectrum2_data.shape[0],
        mz_tolerance,
        score_algorithm=score_algorithm)
    
    result = np.asarray(arr_from_score_vector(matches))
    
    if scoring in ('entropy', 'weighted_entropy'):
        result['score'] /= 2
        
    return result


def cosine_score(double spectrum1_mz,
                 np.ndarray[np.float32_t, ndim=2] spectrum1_data,
                 double spectrum2_mz,
                 np.ndarray[np.float32_t, ndim=2] spectrum2_data,
                 double mz_tolerance,
                 int min_matched_peaks) -> float:   
    return generic_score_nogil(
        spectrum1_mz,
        <peak_t*>np_arr_pointer(spectrum1_data),
        spectrum1_data.shape[0],
        spectrum2_mz,
        <peak_t*>np_arr_pointer(spectrum2_data),
        spectrum2_data.shape[0],
        mz_tolerance, min_matched_peaks,
        score_algorithm_t.cosine)

    
def weighted_entropy_score(double spectrum1_mz, np.ndarray[np.float32_t, ndim=2] spectrum1_data,
                             double spectrum2_mz,np.ndarray[np.float32_t, ndim=2] spectrum2_data,
                             double mz_tolerance, int min_matched_peaks) -> float:   
    return generic_score_nogil(
        spectrum1_mz,
        <peak_t*>np_arr_pointer(spectrum1_data),
        spectrum1_data.shape[0],
        spectrum2_mz,
        <peak_t*>np_arr_pointer(spectrum2_data),
        spectrum2_data.shape[0],
        mz_tolerance, min_matched_peaks,
        score_algorithm_t.weighted_entropy)
        

def entropy_score(double spectrum1_mz, np.ndarray[np.float32_t, ndim=2] spectrum1_data,
                  double spectrum2_mz, np.ndarray[np.float32_t, ndim=2] spectrum2_data,
                  double mz_tolerance, int min_matched_peaks) -> float:
    return generic_score_nogil(
        spectrum1_mz,
        <peak_t*>np_arr_pointer(spectrum1_data),
        spectrum1_data.shape[0],
        spectrum2_mz,
        <peak_t*>np_arr_pointer(spectrum2_data),
        spectrum2_data.shape[0],
        mz_tolerance, min_matched_peaks,
        score_algorithm_t.entropy)

       
def apply_weight_to_intensity(np.ndarray[np.float32_t, ndim=2] spectrum_data):
    cdef:
        vector[peak_t] peaks
        np.ndarray[np.float32_t, ndim=2] weighted
        
    if spectrum_data.shape[0] == 0:
        return peaks

    peaks = apply_weight_to_intensity_no_gil(<peak_t*>np_arr_pointer(spectrum_data), spectrum_data.shape[0])
      
    weighted = arr_from_peaks_vector(peaks)
    
    # Free memory
    peaks.clear()
    peaks.shrink_to_fit()
    
    return weighted


def spectral_entropy(np.ndarray[np.float32_t, ndim=2] spectrum_data):
    return spectral_entropy_no_gil(<peak_t*>np_arr_pointer(spectrum_data), spectrum_data.shape[0])

    
def compute_similarity_matrix(vector[double] mzvec, list datavec,
                              double mz_tolerance, int min_matched_peaks,
                              score='cosine',
                              object callback=None, dense_output=True):
    cdef:
        np.ndarray[np.float32_t, ndim=2] tmp_array
        size_t size = mzvec.size()
        int i
        vector[peak_t *] data_p
        vector[np.npy_intp] data_sizes
        score_algorithm_t score_algorithm = str_to_score_algorithm(score)
        
    data_p.resize(size)
    data_sizes.resize(size)
    for i, tmp_array in enumerate(datavec):
        data_p[i] = <peak_t*>np_arr_pointer(tmp_array)
        data_sizes[i] = tmp_array.shape[0]
        
    if dense_output:
        matrix = np.asarray(compute_similarity_matrix_dense_nogil(mzvec, data_p, data_sizes,
                                                                  mz_tolerance, min_matched_peaks,
                                                                  score_algorithm, callback))
        matrix[matrix>1] = 1
    else:
        data, col_ind, row_ind = compute_similarity_matrix_sparse_nogil(mzvec, data_p, data_sizes,
                                                                        mz_tolerance, min_matched_peaks,
                                                                        score_algorithm, callback)
        matrix = csr_matrix((arr_from_1d_vector(data, np.dtype(np.float32)),
                             (arr_from_1d_vector(col_ind, np.dtype(np.int32)),
                             arr_from_1d_vector(row_ind, np.dtype(np.int32)))),
                            dtype=np.float32)

    data_sizes.clear()
    data_sizes.shrink_to_fit()
    return matrix
