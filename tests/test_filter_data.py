"""
Test `libmetgem.filter.filter_data`.
"""

import pytest
import numpy as np

from libmetgem import IS_CYTHONIZED, MZ, INTENSITY
from libmetgem.filter import filter_data


from data import (random_spectrum, known_spectrum_filter_comparison,
                  min_intensity, parent_filter_tolerance,
                  matched_peaks_window, min_matched_peaks_search)


def test_filter_data_known(known_spectrum_filter_comparison):
    """Test against known results.
    """
    
    parent, data, expected = known_spectrum_filter_comparison
     
    expected = np.sort(expected, axis=0)
    filtered = np.sort(filter_data(parent, data, 0, 17, 50, 6), axis=0)
       
    assert filtered == pytest.approx(expected)
    
    
def test_filter_data_already_filtered(random_spectrum, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search):
    """If filtering an already filtered array with the same parameters,
       *m/z* values should not change.
    """

    parent, data = random_spectrum
    data = filter_data(parent, data, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search)
    
    expected = np.sort(data, axis=0)
    filtered = np.sort(filter_data(parent, data, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search), axis=0)
    
    assert filtered[:, MZ] == pytest.approx(expected[:, MZ])
    
    
def test_filter_data_norm(random_spectrum, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search):
    """Norm of filtered data should always be 1"""
    
    parent, data = random_spectrum
    
    data = data.copy()
    data[:, INTENSITY] = data[:, INTENSITY] * 100
    
    filtered = np.sort(filter_data(parent, data, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search), axis=0)
    
    if filtered.size > 0:
        assert pytest.approx(filtered[:, INTENSITY] @ filtered[:, INTENSITY]) == 1.
    
    
def test_filter_data_no_filtering(random_spectrum):
    """If all parameters are set to zero, no *m/z* should be filtered.
    """
    
    parent, data = random_spectrum
    expected = np.sort(data, axis=0)
    filtered = np.sort(filter_data(parent, data, 0, 0, 0, 0), axis=0)
    
    assert filtered.shape == expected.shape
    assert filtered[:, MZ] == pytest.approx(expected[:, MZ], rel=1e-4)
    
    
def test_filter_data_low_mass(random_spectrum, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search):
    """Low mass peaks should be filtered out"""
    
    parent, data = random_spectrum
    
    # Make sure we have *m/z* below 50
    data = data.copy()
    data[0, MZ] = np.random.random((1,)) * 50
    
    filtered = np.sort(filter_data(parent, data, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search), axis=0)
    
    assert filtered.shape < data.shape
    if filtered.size > 0:
        assert filtered[:, MZ].min() > 50
        for mz in filtered[:, MZ]:
            assert mz > 50
    

@pytest.mark.parametrize("parent_filter_tolerance", [0, 17, 5, 4, 20, 50])    
def test_filter_data_parent(random_spectrum, parent_filter_tolerance):
    """Peaks close to the parent mass should be filtered out"""
    
    parent, data = random_spectrum
    
    # Make sure we have *m/z* in the parent+/-parent_filter_tolerance range
    data = data.copy()
    data[0, MZ] = parent + np.random.random((1,)) * parent_filter_tolerance
    # Make sure that excluding range is strict
    data[1, MZ] = parent + parent_filter_tolerance
    data[2, MZ] = parent - parent_filter_tolerance
    
    filtered = np.sort(filter_data(parent, data, 0, parent_filter_tolerance,
                                   50, 6), axis=0)
                                   
    if parent_filter_tolerance == 0:
        assert filtered.shape == data.shape
    elif parent_filter_tolerance > 0:
        assert filtered.shape < data.shape
    
    for mz in filtered[:, MZ]:
        assert mz<parent-parent_filter_tolerance \
               or mz == pytest.approx(parent-parent_filter_tolerance) \
               or mz>parent+parent_filter_tolerance \
               or mz == pytest.approx(parent+parent_filter_tolerance)
               
               
               
@pytest.mark.parametrize("min_intensity", [0, 10, 100, 101, 200])
def test_filter_data_min_intensity(random_spectrum, min_intensity):
    """Peaks higher than `min_intensity` % of maximum intensity should be
       filtered out"""
    
    parent, data = random_spectrum
    
    filtered = np.sort(filter_data(parent, data, min_intensity, 0,
                                   0, 0), axis=0)
    
    if min_intensity == 0:
        assert filtered.shape == data.shape
        assert filtered[:, MZ] == pytest.approx(np.sort(data[:, MZ], axis=0))
    elif min_intensity == 100:
        assert filtered.shape[0] == 1
    elif min_intensity > 100:
        assert filtered.shape[0] == 0
    else:
        assert filtered.shape <= data.shape
        max = filtered[:, INTENSITY].max()
        for intensity in filtered[:, INTENSITY]:
            assert intensity <= max
            
            
@pytest.mark.parametrize("matched_peaks_window", [0, 10, 50, 100])
@pytest.mark.parametrize("min_matched_peaks_search", [0, 6, 12, 18, 24])
def test_filter_data_window(random_spectrum, matched_peaks_window,
                            min_matched_peaks_search):
    """If `matched_peaks_window` or `min_matched_peaks_search` is zero,
       no peaks should be filtered.
    """
    
    parent, data = random_spectrum
    
    filtered = np.sort(filter_data(parent, data, 0, 0, matched_peaks_window,
                                   min_matched_peaks_search), axis=0)
    
    if matched_peaks_window == 0 or min_matched_peaks_search == 0:
        assert filtered.shape == data.shape
        assert filtered[:, MZ] == pytest.approx(np.sort(data[:, MZ], axis=0))
    else:
        assert filtered.shape <= data.shape
            
            
def test_filter_data_non_contiguous(random_spectrum, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search):
    """f-contiguous arrays should give same results than c-contiguous arrays.
    """
    
    parent, data = random_spectrum
    
    filtered = np.sort(filter_data(parent, data, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search), axis=0)
    data = np.asfortranarray(data, dtype=data.dtype)
    filtered_nc = np.sort(filter_data(parent, data, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search), axis=0)
    
    assert filtered == pytest.approx(filtered_nc)
    
    
def test_filter_data_reversed(random_spectrum, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search):
    """Reversed arrays should give same results than non-reversed arrays.
    """
    
    parent, data = random_spectrum
    
    filtered = np.sort(filter_data(parent, data, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search), axis=0)
    filtered_r = np.sort(filter_data(parent, data[::-1], min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search), axis=0)
    
    assert filtered == pytest.approx(filtered_r)
    
    
def test_filter_data_empty(min_intensity, parent_filter_tolerance,
                           matched_peaks_window, min_matched_peaks_search):
    
    parent = 152.569
    data = np.empty((0, 2), dtype=np.float32)
    
    filtered = filter_data(parent, data, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search)
    assert filtered.size == 0

@pytest.mark.python            
@pytest.mark.skipif(not IS_CYTHONIZED, reason="libmetgem should be cythonized")
def test_filter_data_python_cython(random_spectrum, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search):
    """Cythonized `filter_data` and it's fallback Python version should give the
       same results.
    """
    
    parent, data = random_spectrum
    
    filtered_p = np.sort(filter_data.__wrapped__(parent, data, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search), axis=0)
    filtered_c = np.sort(filter_data(parent, data, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search), axis=0)
    
    assert filtered_p.shape == filtered_c.shape
    assert filtered_p == pytest.approx(filtered_c)
