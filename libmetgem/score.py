"""
    Compute cosine similarity score between mass spectra.
"""

from .common import MZ, INTENSITY
from ._loader import load_cython

import warnings
from typing import List, Callable, Tuple, Union
from enum import IntEnum
import numpy as np
import operator
from scipy.sparse import csr_matrix

__all__ = ('cosine_score', 'compare_spectra',
           'compute_distance_matrix', 'compute_similarity_matrix',
           'SpectraMatchState')

@load_cython
class SpectraMatchState(IntEnum):
    fragment = 0
    neutral_loss = 1


def _partial_score(intensity1: float, intensity2: float, score_algorithm:str = 'cosine') -> float:
    if score_algorithm in ('entropy', 'weighted_entropy'):
        intensity_sum = intensity1 + intensity2
        score = intensity_sum * np.log2(intensity_sum) - intensity1 * np.log2(intensity1) - intensity2 * np.log2(intensity2)
    else:
        score = intensity1 * intensity2
        
    return score

  
def generic_score(spectrum1_mz: float,
                  spectrum1_data: np.ndarray,
                  spectrum2_mz: float,
                  spectrum2_data: np.ndarray,
                  mz_tolerance: float,
                  min_matched_peaks: int,
                  score_algorithm: str = 'cosine'
                  ) -> float:
    
    score, num_matched_peaks, _ = _compare_spectra(
            spectrum1_mz,
            spectrum1_data,
            spectrum2_mz,
            spectrum2_data,
            mz_tolerance,
            score_algorithm = score_algorithm,
            return_matches = False)
                
    if num_matched_peaks < min_matched_peaks:
        return 0.
    
    if score_algorithm in ('entropy', 'weighted_entropy'):
        score /= 2
    return score

  
def _compare_spectra(spectrum1_mz: float, spectrum1_data: np.ndarray,
                     spectrum2_mz: float, spectrum2_data: np.ndarray,
                     mz_tolerance: float, score_algorithm: str ='cosine',
                     return_matches: bool=False) -> Tuple[float, int, List[Tuple[int, int, int]]]:
    if spectrum1_data.shape[MZ] == 0 or spectrum2_data.shape[MZ] == 0:
        return (0., 0, [])
                 
    dm = spectrum1_mz - spectrum2_mz
    
    if score_algorithm == 'weighted_entropy':
        # Apply the weights to the peaks.
        apply_weight_func = getattr(apply_weight_to_intensity, '__wrapped__', apply_weight_to_intensity)
        spectrum1_data = apply_weight_func(spectrum1_data)
        spectrum2_data = apply_weight_func(spectrum2_data)
    
    scores = []
    
    if dm == 0.:
        for i in range(spectrum1_data.shape[0]):
            for j in range(spectrum2_data.shape[0]):
                diff = spectrum2_data[j, MZ] - spectrum1_data[i, MZ]
                if abs(diff) <= mz_tolerance:
                    scores.append((_partial_score(spectrum1_data[i, INTENSITY], spectrum2_data[j, INTENSITY], score_algorithm),
                                   i, j, SpectraMatchState.fragment))
    else:
        for i in range(spectrum1_data.shape[0]):
            for j in range(spectrum2_data.shape[0]):
                diff = spectrum2_data[j, MZ] - spectrum1_data[i, MZ]
                if abs(diff) <= mz_tolerance:
                    scores.append((_partial_score(spectrum1_data[i, INTENSITY], spectrum2_data[j, INTENSITY], score_algorithm),
                                   i, j, SpectraMatchState.fragment))
                elif abs(diff + dm) <= mz_tolerance:
                    scores.append((_partial_score(spectrum1_data[i, INTENSITY], spectrum2_data[j, INTENSITY], score_algorithm),
                                   i, j, SpectraMatchState.neutral_loss))

    if not scores:
        return (0., 0, scores)
        
    scores.sort(reverse=True)
    
    peak_used1 = [False] * spectrum1_data.shape[0]
    peak_used2 = [False] * spectrum2_data.shape[0]

    num_matched_peaks = 0
    total = 0.
    matches = []
    for score, ix1, ix2, t in scores:
        if not peak_used1[ix1] and not peak_used2[ix2]:
            total += score
            peak_used1[ix1] = peak_used2[ix2] = 1
            num_matched_peaks += 1
            if return_matches:
                matches.append((ix1, ix2, score, t))

    return total, num_matched_peaks, matches


@load_cython
def spectral_entropy(spectrum_data: np.ndarray) -> float:
    if spectrum_data.shape[0] == 0:
        return 0.
    else:
        return -np.sum(spectrum_data[:, INTENSITY] * np.log(spectrum_data[:, INTENSITY]))


@load_cython
def apply_weight_to_intensity(spectrum_data: np.ndarray) -> np.ndarray:
    """
    Apply a weight to the intensity of a spectrum based on spectral entropy based on the method described in:

    Li, Y., Kind, T., Folz, J. et al. Spectral entropy outperforms MS/MS dot product similarity for small-molecule compound identification. Nat Methods 18, 1524-1531 (2021). https://doi.org/10.1038/s41592-021-01331-z.

    Parameters
    ----------
    peaks : np.ndarray in shape (n_peaks, 2), np.float32
        The spectrum to apply weight to. The first column is m/z, and the second column is intensity.
        The peaks need to be pre-cleaned.

        _

    Returns
    -------
    np.ndarray in shape (n_peaks, 2), np.float32
        The spectrum with weight applied. The first column is m/z, and the second column is intensity.
        The peaks will be a copy of the input peaks.
    """
    if spectrum_data.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)

    # Calculate the spectral entropy.
    entropy = 0.
    if spectrum_data.shape[0] > 0:
        spectral_entropy_func = getattr(spectral_entropy, '__wrapped__', spectral_entropy)    
        entropy = spectral_entropy_func(spectrum_data)

    # Copy the peaks.
    weighted_data = spectrum_data.copy()

    # Apply the weight.
    if entropy < 3:
        weight = 0.25 + 0.25 * entropy
        weighted_data[:, INTENSITY] = np.power(spectrum_data[:, INTENSITY], weight)
        intensity_sum = np.sum(weighted_data[:, INTENSITY])
        weighted_data[:, INTENSITY] /= intensity_sum

    return weighted_data
    

@load_cython
def cosine_score(
    spectrum1_mz: float, spectrum1_data: np.ndarray,
    spectrum2_mz: float, spectrum2_data: np.ndarray,
    mz_tolerance: float, min_matched_peaks: float) -> float:
    """
        Compute cosine similarity score of two spectra.

    Args:
        spectrum1_mz: *m/z* value of first spectrum.
        spectrum1_data: 2D array containing first spectrum data. Intensities
            should be normalized to norm 1.
        spectrum2_mz: *m/z* value of first spectrum.
        spectrum2_data: 2D array containing first spectrum data. Intensities
            should be normalized to norm 1.
        mz_tolerance: maximum *m/z* delta allowed between two peaks to consider
            them as identical.
        min_matched_peaks: score will be 0 if the two spectra have less than
            `min_matched_peaks` peaks in common.
    
    Returns:
        Cosine similarity between spectrum1 and spectrum2.
    """
    
    return generic_score(spectrum1_mz, spectrum1_data,
                         spectrum2_mz, spectrum2_data,
                         mz_tolerance, min_matched_peaks,
                         score_algorithm='cosine')


@load_cython
def weighted_entropy_score(
    spectrum1_mz: float, spectrum1_data: np.ndarray,
    spectrum2_mz: float, spectrum2_data: np.ndarray,
    mz_tolerance: float, min_matched_peaks: float) -> float:
    
    return generic_score(spectrum1_mz, spectrum1_data,
                     spectrum2_mz, spectrum2_data,
                     mz_tolerance, min_matched_peaks,
                     score_algorithm='weighted_entropy')

 
@load_cython
def entropy_score(
    spectrum1_mz: float, spectrum1_data: np.ndarray,
    spectrum2_mz: float, spectrum2_data: np.ndarray,
    mz_tolerance: float, min_matched_peaks: float) -> float:
        
    return generic_score(spectrum1_mz, spectrum1_data,
                 spectrum2_mz, spectrum2_data,
                 mz_tolerance, min_matched_peaks,
                 score_algorithm='entropy')


@load_cython
def compare_spectra(spectrum1_mz: float, spectrum1_data: np.ndarray,
                    spectrum2_mz: float, spectrum2_data: np.ndarray,
                    mz_tolerance: float, scoring: str = 'cosine') -> np.ndarray:
    """
        Compute similarity score of two spectra and return an array
        of indexes of matches peaks from the two spectra.

    Args:
        spectrum1_mz: *m/z* value of first spectrum.
        spectrum1_data: 2D array containing first spectrum data. Intensities
            should be normalized to norm 1.
        spectrum2_mz: *m/z* value of first spectrum.
        spectrum2_data: 2D array containing first spectrum data. Intensities
            should be normalized to norm 1.
        mz_tolerance: maximum *m/z* delta allowed between two peaks to consider
            them as identical.
        scoring: Algorithm used for spectral comparison.
    
    Returns:
        Record array with four columns:
            ix1 -> indexes of peaks from first spectrum,
            ix2 -> indexes of peaks from second spectrum,
            score -> partial score,
            type -> type of match (fragment or neutral loss)
    """
    
    _, _, matches = _compare_spectra(spectrum1_mz, spectrum1_data,
                                     spectrum2_mz, spectrum2_data,
                                     mz_tolerance, scoring,
                                     return_matches=True)
    result = np.asarray(matches, dtype=np.dtype([('ix1', '<u2'), ('ix2', '<u2'), ('score', '<f8'), ('type', '<u1')]))
    
    if scoring in ('entropy', 'weighted_entropy'):
        result['score'] /= 2
    
    return result


@load_cython
def compute_similarity_matrix(mzs: List[float], spectra: List[np.ndarray],
                            mz_tolerance: float, min_matched_peaks: float,
                            scoring: str = 'cosine',
                            callback: Callable[[int], bool]=None, dense_output: bool=True) -> Union[np.ndarray, csr_matrix]:
    """
        Compute pairwise similarity matrix of a list of spectra.
    
    Args:
        mzs: list of *m/z* values.
        spectra: list of 2D array containing spectra data.
        mz_tolerance: maximum *m/z* delta allowed between two peaks to consider
            them as identical.
        min_matched_peaks: score will be 0 if the two spectra have less than
            `min_matched_peaks` peaks in common.
        scoring: Algorithm used for spectral comparison.
        callback: function called to track progress of computation. First
            parameter (`int`) is the number of spectra computed since last call.
            It should return True if processing should continue, or False if
            computations should stop.
        dense_output: whether the returned value should be a csr sparse matrix
            or a dense numpy array.
            
    Returns:
        Pairwise similarity matrix of the given spectra.
    
    See Also:
        cosine_score     
    """
    
    size = len(mzs)
    if dense_output:
        matrix = np.empty((size, size), dtype=np.float32)
        for i in range(size):
            for j in range(i):
                matrix[i, j] = matrix[j, i] = generic_score(mzs[i], spectra[i], mzs[j], spectra[j],
                                                            mz_tolerance, min_matched_peaks,
                                                            scoring)
            if callback is not None:
                if not callback(i):
                    return
        np.fill_diagonal(matrix, 1)
        matrix[matrix > 1] = 1
    
        return matrix
    else:
        data = []
        indices = []
        indptr = []
        count = 0
        for i in range(size):
            data.append(1)
            indices.append(i)
            indptr.append(count)
            count += 1
            for j in range(i+1, size):
                score = generic_score(mzs[i], spectra[i], mzs[j], spectra[j],
                                      mz_tolerance, min_matched_peaks,
                                      scoring)
                if score > 0:
                    data.append(min(score, 1))
                    indices.append(j)
                    count += 1
                if callback is not None:
                    if not callback(i):
                        return
        indptr.append(count)
        
        data = np.asarray(data, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.int32)
        indptr = np.asarray(indptr, dtype=np.int32)
        matrix = csr_matrix((data, indices, indptr), dtype=np.float32)
        matrix = matrix + matrix.T 
        matrix.setdiag(1)
        return matrix
