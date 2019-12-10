"""
    Compute cosine similarity score between mass spectra.
"""

from .common import MZ, INTENSITY
from ._loader import load_cython

from typing import List, Callable, Tuple
from enum import Enum
import numpy as np
import operator

__all__ = ('cosine_score', 'compare_spectra', 'compute_distance_matrix', 'SpectraMatchState')

@load_cython
class SpectraMatchState(Enum):
    fragment = 0
    neutral_loss = 1
    
def _compare_spectra(spectrum1_mz: float, spectrum1_data: np.ndarray,
                 spectrum2_mz: float, spectrum2_data: np.ndarray,
                 mz_tolerance: float, return_matches: bool=False) -> Tuple[float, int, List[Tuple[int, int, int]]]:
    if spectrum1_data.shape[MZ] == 0 or spectrum2_data.shape[MZ] == 0:
        return (0., 0, [])
                 
    dm = spectrum1_mz - spectrum2_mz
    
    scores = []
    
    if dm == 0.:
        for i in range(spectrum1_data.shape[0]):
            for j in range(spectrum2_data.shape[0]):
                diff = spectrum2_data[j, MZ] - spectrum1_data[i, MZ]
                if abs(diff) <= mz_tolerance:
                    scores.append((spectrum1_data[i, INTENSITY] * spectrum2_data[j, INTENSITY], i, j, SpectraMatchState.fragment))
    else:
        for i in range(spectrum1_data.shape[0]):
            for j in range(spectrum2_data.shape[0]):
                diff = spectrum2_data[j, MZ] - spectrum1_data[i, MZ]
                if abs(diff) <= mz_tolerance:
                    scores.append((spectrum1_data[i, INTENSITY] * spectrum2_data[j, INTENSITY], i, j, SpectraMatchState.fragment))
                elif abs(diff + dm) <= mz_tolerance:
                    scores.append((spectrum1_data[i, INTENSITY] * spectrum2_data[j, INTENSITY], i, j, SpectraMatchState.neutral_loss))  

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
                matches.append((ix1, ix2, t))

    return total, num_matched_peaks, matches

@load_cython
def cosine_score(spectrum1_mz: float, spectrum1_data: np.ndarray,
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
    score, num_matched_peaks, _ = _compare_spectra(spectrum1_mz, spectrum1_data,
                                                   spectrum2_mz, spectrum2_data,
                                                   mz_tolerance, return_matches=False)
    if num_matched_peaks < min_matched_peaks:
        return 0.
        
    return score
    
@load_cython
def compare_spectra(spectrum1_mz: float, spectrum1_data: np.ndarray,
                    spectrum2_mz: float, spectrum2_data: np.ndarray,
                    mz_tolerance: float) -> np.ndarray:
    """
        Compute cosine similarity score of two spectra and return an array
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
    
    Returns:
        Array with three columns:
            1. indexes of peaks from first spectrum,
            2. indexes of peaks from second spectrum,
            3. type of match (fragment or neutral loss)
    """
    _, _, matches = _compare_spectra(spectrum1_mz, spectrum1_data,
                                     spectrum2_mz, spectrum2_data,
                                     mz_tolerance, return_matches=True)
    return np.asarray(matches, dtype=np.uint16)
    
@load_cython
def compute_distance_matrix(mzs: List[float], spectra: List[np.ndarray],
                            mz_tolerance: float, min_matched_peaks: float,
                            callback: Callable[[int], bool]=None) -> np.ndarray:
    """
        Compute pairwise distance matrix of a list of spectra.
    
    Args:
        mzs: list of *m/z* values.
        spectra: list of 2D array containing spectra data.
        mz_tolerance: maximum *m/z* delta allowed between two peaks to consider
            them as identical.
        min_matched_peaks: score will be 0 if the two spectra have less than
            `min_matched_peaks` peaks in common.
        callback: function called to track progress of computation. First
            parameter (`int`) is the number of spectra computed since last call.
            It should return True if processing should continue, or False if
            computations should stop.
            
    Returns:
        Pairwise distance matrix of the given spectra.
    
    See Also:
        cosine_score     
    """
    
    size = len(mzs)
    matrix = np.empty((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(i):
            matrix[i, j] = matrix[j, i] = cosine_score(mzs[i], spectra[i], mzs[j], spectra[j],
                                                       mz_tolerance, min_matched_peaks)
        if callback is not None:
            if not callback(i-1):
                return matrix
    np.fill_diagonal(matrix, 1)
    matrix[matrix > 1] = 1
    return matrix