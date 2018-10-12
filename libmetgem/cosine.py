"""
    Compute cosine similarity score between mass spectra.
"""

from .common import MZ
from ._loader import load_cython

from typing import List, Callable
import numpy as np
import operator

__all__ = ('cosine_score', 'compute_distance_matrix')

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
    
    dm = spectrum1_mz - spectrum2_mz

    diff_matrix = spectrum2_data[:, MZ] - spectrum1_data[:, MZ][:, None]
    if dm != 0.:
        idxMatched1, idxMatched2 = np.where(
            np.logical_or(np.abs(diff_matrix) <= mz_tolerance,
                          np.abs(diff_matrix + dm) <= mz_tolerance))
    else:
        idxMatched1, idxMatched2 = np.where(np.abs(diff_matrix) <= mz_tolerance)
    del diff_matrix

    if idxMatched1.size + idxMatched2.size == 0:
        return 0.

    peakUsed1 = [False] * spectrum1_data.size
    peakUsed2 = [False] * spectrum2_data.size

    peakMatches = []
    for i in range(idxMatched1.size):
        peakMatches.append((idxMatched1[i], idxMatched2[i],
                            spectrum1_data[idxMatched1[i], 1] * spectrum2_data[
                                idxMatched2[i], 1]))

    peakMatches = sorted(peakMatches, key=operator.itemgetter(2), reverse=True)

    score = 0.
    numMatchedPeaks = 0
    for i in range(len(peakMatches)):
        if not peakUsed1[peakMatches[i][0]] and not peakUsed2[peakMatches[i][1]]:
            score += peakMatches[i][2]
            peakUsed1[peakMatches[i][0]] = True
            peakUsed2[peakMatches[i][1]] = True
            numMatchedPeaks += 1

    if numMatchedPeaks < min_matched_peaks:
        return 0.

    return score

    
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