"""
    Filter MS/MS spectra before cosine score calculations.
"""

from .common import MZ, INTENSITY
from ._loader import load_cython

from typing import List, Callable

import numpy as np

__all__ = ('filter_data', 'filter_data_multi')


def parent_filter(mz_parent: float, data: np.ndarray, min_intensity: int,
                parent_filter_tolerance: int, mz_min: float = 50.) -> np.ndarray:
    # Filter low mass peaks
    data = data[data[:, MZ] >= mz_min]

    # Filter peaks close to the parent ion's m/z
    data = data[np.logical_or(data[:, MZ] <= mz_parent - parent_filter_tolerance,
                              data[:, MZ] >= mz_parent + parent_filter_tolerance)]

    if data.size > 0:
        # Keep only peaks higher than threshold
        data = data[data[:, INTENSITY] >= min_intensity / 100. * data[:, INTENSITY].max()]
        
    return data
    
    
def window_rank_filter(data: np.ndarray,
                       matched_peaks_window: float,
                       min_matched_peaks_search: float) -> np.ndarray:
    if data.size > 0:
        # Window rank filter
        data = data[np.argsort(data[:, INTENSITY])]

        if data.size > 0:
            mz_ratios = data[:, MZ]
            mask = np.logical_and(mz_ratios >= mz_ratios[:, None] - matched_peaks_window,
                                  mz_ratios <= mz_ratios[:, None] + matched_peaks_window)
            data = data[np.array([mz_ratios[i] in mz_ratios[mask[i]][-min_matched_peaks_search:]
                                  for i in range(mask.shape[0])])]

    return data
    

@load_cython
def square_root_data(data: np.ndarray, copy: bool=True) -> np.ndarray:
    """
        Replace intensities of an MS/MS spectrum with their square-root.
        
    Args:
        data: A 2D array representing an MS/MS spectrum.
        
    Returns:
        A copy of the array with intensities square-rooted.
        
    See Also:
        filter_data
    """
    
    if data.size == 0:
        return data
        
    if copy:
        data = data.copy()
    
    # Use square root of intensities to minimize/maximize effects of high/low intensity peaks
    data[:, INTENSITY] = np.sqrt(data[:, INTENSITY]) * 10
    
    return data
    
    
@load_cython
def normalize_data(data: np.ndarray, norm='dot', copy: bool=True) -> np.ndarray:
    """
        Normalize intensities of an MS/MS spectrum to norm 1.
        
    Args:
        data: A 2D array representing an MS/MS spectrum.
        norm: A string reprensenting normalization method.
            Can be 'dot' for dot product normalization
            or 'sum' to normalize the intensities to sum to 1.
        
    Returns:
        A copy of the array with intensities normalised.
        
    See Also:
        filter_data
    """
    
    if data.size == 0:
        return data
        
    if copy:
        data = data.copy()
    
    if norm == 'sum':
        # Normalize the intensity to sum to 1
        data[:, INTENSITY] = data[:, INTENSITY] / data[:, INTENSITY].sum()
    else:
        # Normalize data to norm 1
        data[:, INTENSITY] = data[:, INTENSITY] / np.sqrt(data[:, INTENSITY] @ data[:, INTENSITY])
    
    return data


@load_cython
def filter_data(mz_parent: float, data: np.ndarray, min_intensity: int,
                parent_filter_tolerance: int, matched_peaks_window: float,
                min_matched_peaks_search: float,
                mz_min: float = 50.,
                square_root: bool=True, norm: str='dot') -> np.ndarray:
    """
        6-step filter of an array representing an MS/MS spectrum:
            * Low mass filtering: Remove all peaks with *m/z* lower than `mz_min`.
            * Parent filtering: Remove peaks with *m/z* in the closed interval
              `mz_parent` +/- `parent_filter_tolerance`.
            * Threshold filtering: Keep only peaks with an intensity higher than
              or equal to `min_intensity` % of the maximum intensity.
            * Window rank filtering: For each peak in the spectrum, remove it if
              it is not one of the `min_matched_peaks_search` highest peaks in
              the `matched_peaks_window` window.
            * Square-root: Replace each peak intensity by it's square-root to
              minimize influence of highest intensities and maximize influence
              of lowest intensities.
            * Normalization: Normalize intensities to norm 1.
        
    Args:
        mz_parent: *m/z* of parent ion. Used for `parent filtering` step.
        data: 2D array containing spectrum data.
        min_intensity: Relative intensity in percentage of maximum. Used for
            `threshold filtering` step.
        parent_filter_tolerance: Control size of the excluding range around
            *m/z* of parent ion. Used for `parent filtering` step.
        matched_peaks_window: Control window's size used in the
            `window rank filtering` step.
        min_matched_peaks_search: Control how many peaks to keep at each step
            during the `window rank filtering` step.
        mz_min: All peaks with *m/z* below this value will be filtered out.
        square_root: Whether the intensities should be square-rooted.
        norm: A string reprensenting normalization method.
            Can be 'dot' for dot product normalization
            or 'sum' to normalize the intensities to sum to 1.
    
    Returns:
        A filtered array.

    """
    
    if data.size == 0:
        return data

    if min_intensity > 0 or parent_filter_tolerance > 0 or min(data[:,MZ]) < mz_min:
        data = parent_filter(mz_parent, data, min_intensity, parent_filter_tolerance, mz_min)
        
    if matched_peaks_window > 0 and min_matched_peaks_search > 0:
        data = window_rank_filter(data, matched_peaks_window, min_matched_peaks_search)
        
    if square_root:
        square_func = getattr(square_root_data, '__wrapped__', square_root_data)
        data = square_func(data)
    
    norm_func = getattr(normalize_data, '__wrapped__', normalize_data)
    data = norm_func(data, norm, copy = True)
    
    return data


@load_cython    
def filter_data_multi(mzvec: List[float], datavec: List[np.ndarray],
                      min_intensity: int, parent_filter_tolerance: int,
                      matched_peaks_window: float,
                      min_matched_peaks_search: float,
                      mz_min: float = 50.,
                      square_root: bool=True, norm: str='dot',
                      callback: Callable[[int], bool]=None) -> List[np.ndarray]:
                      
    """
        Filter a list of MS/MS spectra.
        
    Args:
        mzvec: List of parent ions' *m/z*.
        datavec: List of 2D arrays containing spectra data.
        min_intensity: Relative intensity in percentage of maximum.
        parent_filter_tolerance: Control size of the excluding range around
            *m/z* of parent ion.
        matched_peaks_window: Control window's size used in the
            `window rank filtering` step.
        min_matched_peaks_search: Control how many peaks to keep at each step
            during the `window rank filtering` step.
        mz_min: All peaks with *m/z* below this value will be filtered out.
        square_root: Whether the intensities should be square-rooted.
        norm: A string reprensenting normalization method.
            Can be 'dot' for dot product normalization
            or 'sum' to normalize the intensities to sum to 1.
        callback: function called to track progress of computation. First
            parameter (`int`) is the number of spectra computed since last call.
            It should return True if processing should continue, or False if
            computations should stop.
    Returns:
        A filtered array.
        
    
    See Also:
        filter_data
    
    """
                      
    size = len(mzvec)
    has_callback = callback is not None
    result = []

    for i in range(size):
        result.append(filter_data(mzvec[i], datavec[i], min_intensity, parent_filter_tolerance,
                                  matched_peaks_window, min_matched_peaks_search, mz_min,
                                  square_root, norm))
        if has_callback and i % 10 == 0:
            callback(10)

    if has_callback and size % 10 != 0:
        callback(size % 10)

    return result
