"""
Test `libmetgem.filter.filter_data_multi`.
"""

import pytest
import numpy as np

from libmetgem import IS_CYTHONIZED, MZ, INTENSITY
from libmetgem.filter import filter_data_multi
from funcs import filter_data_multi_f

from data import (random_spectra, known_spectra_filter_comparison,
                  min_intensity, parent_filter_tolerance,
                  matched_peaks_window, min_matched_peaks_search,
                  mz_min)

                  
def test_filter_data_multi_known(known_spectra_filter_comparison, filter_data_multi_f):
    norm, mzs, unfiltered_spectra, spectra = known_spectra_filter_comparison
    
    filtered_spectra = filter_data_multi_f(mzs, unfiltered_spectra,
                                         0, 17, 50, 6, 50,
                                         square_root=norm=='dot',
                                         norm=norm)
    
    for data, expected in zip(filtered_spectra, spectra):
        assert np.sort(data, axis=0) == pytest.approx(np.sort(expected, axis=0))
        

@pytest.mark.python
@pytest.mark.skipif(not IS_CYTHONIZED, reason="libmetgem should be cythonized")
def test_filter_data_multi_python_cython(scoring, random_spectra,
                min_intensity, parent_filter_tolerance,
                matched_peaks_window, min_matched_peaks_search,
                mz_min):
    """Cythonized `filter_data` and it's fallback Python version should give the
       same results.
    """
    
    mzs, spectra = random_spectra(scoring)
    
    filtered_p = filter_data_multi.__wrapped__(mzs, spectra,
                                               min_intensity,
                                               parent_filter_tolerance,
                                               matched_peaks_window,
                                               min_matched_peaks_search,
                                               mz_min,
                                               scoring)
    filtered_c = filter_data_multi(mzs, spectra, min_intensity,
                                   parent_filter_tolerance,
                                   matched_peaks_window,
                                   min_matched_peaks_search,
                                   mz_min,
                                   scoring)
    
    for p, c in zip(filtered_p, filtered_c):
        assert np.sort(p, axis=0) == pytest.approx(np.sort(c, axis=0))


def test_filter_data_multi_callback_count(scoring, random_spectra, mocker, filter_data_multi_f):
    """callback shoud be called one times per 10 spectrum."""
        
    callback = mocker.Mock(return_value=True)
    
    mzs, spectra = random_spectra(scoring)
    matrix = filter_data_multi_f(mzs, spectra, 0, 0.02, 50, 6, 50, scoring, callback=callback)
    
    expected_num_calls = len(mzs) % 10 + int(len(mzs) % 10 == 0)
    assert callback.call_count == expected_num_calls
    
def test_filter_data_multi_callback_abort(scoring, random_spectra, mocker, filter_data_multi_f):
    """process should be stopped if callback return False."""
        
    callback = mocker.Mock(return_value=False)
       
    mzs, spectra = random_spectra(scoring)
    matrix = filter_data_multi_f(mzs, spectra, 0, 0.02, 50, 6, 50, scoring, callback=callback)
    
    assert callback.call_count < len(mzs)
