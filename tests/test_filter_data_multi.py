"""
Test `libmetgem.filter.filter_data_multi`.
"""

import pytest
import numpy as np

from libmetgem.common import MZ, INTENSITY
from libmetgem.filter import filter_data_multi

from data import (random_spectra, known_spectra_filter_comparison,
                  min_intensity, parent_filter_tolerance,
                  matched_peaks_window, min_matched_peaks_search)

                  
def test_filter_data_multi_known(known_spectra_filter_comparison,
                min_intensity, parent_filter_tolerance,
                matched_peaks_window, min_matched_peaks_search):
    mzs, unfiltered_spectra, spectra = known_spectra_filter_comparison
    
    filtered_spectra = filter_data_multi(mzs, unfiltered_spectra,
                                         min_intensity,
                                         parent_filter_tolerance,
                                         matched_peaks_window,
                                         min_matched_peaks_search)
    
    for data in filtered_spectra:
        assert np.sort(data, axis=0) == pytest.approx(np.sort(data, axis=0))
        

@pytest.mark.python
@pytest.mark.skipif(getattr(filter_data_multi, '__wrapped__', None) is None,
                    reason="libmetgem should be cythonized")
def test_filter_data_multi_python_cython(random_spectra,
                min_intensity, parent_filter_tolerance,
                matched_peaks_window, min_matched_peaks_search):
    """Cythonized `filter_data` and it's fallback Python version should give the
       same results.
    """
    
    mzs, spectra = random_spectra
    
    filtered_p = filter_data_multi.__wrapped__(mzs, spectra,
                                               min_intensity,
                                               parent_filter_tolerance,
                                               matched_peaks_window,
                                               min_matched_peaks_search)
    filtered_c = filter_data_multi(mzs, spectra, min_intensity,
                                   parent_filter_tolerance,
                                   matched_peaks_window,
                                   min_matched_peaks_search)
    
    for p, c in zip(filtered_p, filtered_c):
        assert np.sort(p, axis=0) == pytest.approx(np.sort(c, axis=0))