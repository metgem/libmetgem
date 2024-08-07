"""
Test `libmetgem.filter.filter_data`.
"""

import pytest
import numpy as np

from libmetgem import IS_CYTHONIZED, MZ, INTENSITY
from libmetgem.filter import filter_data


from data import (random_spectrum, known_spectrum_filter_comparison,
                  min_intensity, parent_filter_tolerance,
                  matched_peaks_window, min_matched_peaks_search,
                  mz_min)


def test_filter_data_known(known_spectrum_filter_comparison):
    """Test against known results.
    """
    
    norm, parent, data, expected = known_spectrum_filter_comparison
     
    expected = np.sort(expected, axis=0)
    filtered = np.sort(filter_data(parent, data, 0, 17, 50, 6, 50,
                                   square_root=True if norm=='dot' else False,
                                   norm=norm),
                       axis=0)
       
    assert filtered == pytest.approx(expected)
    
   
def test_filter_data_already_filtered(scoring, random_spectrum, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search, mz_min):
    """If filtering an already filtered array with the same parameters,
       *m/z* values should not change.
    """

    parent, data = random_spectrum(scoring)
    data = filter_data(parent, data, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search, mz_min,
                  scoring=='cosine', 'dot' if scoring=='cosine' else 'sum')
    
    expected = np.sort(data, axis=0)
    filtered = np.sort(filter_data(parent, data, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search, mz_min, scoring), axis=0)
    
    assert filtered[:, MZ] == pytest.approx(expected[:, MZ])
    

def test_filter_data_norm(scoring, random_spectrum, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search, mz_min):
    """Norm of filtered data should always be 1"""
    
    parent, data = random_spectrum(scoring)
    
    data = data.copy()
    data[:, INTENSITY] = data[:, INTENSITY] * 100
    
    filtered = np.sort(filter_data(parent, data, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search, mz_min,
                  scoring=='cosine', 'dot' if scoring=='cosine' else 'sum'), axis=0)
    
    if filtered.size > 0:
        if scoring == 'cosine':
            assert filtered[:, INTENSITY] @ filtered[:, INTENSITY] == pytest.approx(1.)
        else:
            assert filtered[:, INTENSITY].sum() == pytest.approx(1.)
    
    
def test_filter_data_no_filtering(scoring, random_spectrum):
    """If all parameters are set to zero, no *m/z* should be filtered.
    """
    
    parent, data = random_spectrum(scoring)
    expected = np.sort(data, axis=0)
    filtered = np.sort(filter_data(parent, data, 0, 0, 0, 0, 0,
                        scoring=='cosine', 'dot' if scoring=='cosine' else 'sum'), axis=0)
    
    assert filtered.shape == expected.shape
    assert filtered[:, MZ] == pytest.approx(expected[:, MZ], rel=1e-4)
    

def test_filter_data_low_mass(scoring, random_spectrum, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search, mz_min):
    """Low mass peaks should be filtered out"""
    
    parent, data = random_spectrum(scoring)
    
    # Make sure we have *m/z* below `mz_min`
    data = data.copy()
    data[0, MZ] = np.random.random() * mz_min
    
    filtered = np.sort(filter_data(parent, data, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search, mz_min,
                  scoring=='cosine', 'dot' if scoring=='cosine' else 'sum'), axis=0)
    
    assert filtered.shape < data.shape
    if filtered.size > 0:
        assert filtered[:, MZ].min() > mz_min
        for mz in filtered[:, MZ]:
            assert mz > mz_min


def test_filter_data_low_mass_default_value(scoring, random_spectrum):
    """If not specified, low mass peaks cut off has to be 50"""
    
    parent, data = random_spectrum(scoring)
    
    # Make sure we have *m/z* below 50
    data = data.copy()
    data[0, MZ] = np.random.random() * 50
    
    filtered = np.sort(filter_data(parent, data, 0, 17, 50, 6,
                                   square_root=scoring=='cosine',
                                   norm='dot' if scoring=='cosine' else 'sum'),
                       axis=0)
    
    assert filtered.shape < data.shape
    if filtered.size > 0:
        assert filtered[:, MZ].min() > 50
        for mz in filtered[:, MZ]:
            assert mz > 50
    

@pytest.mark.parametrize("parent_filter_tolerance", [0, 17, 5, 4, 20, 50])    
def test_filter_data_parent(scoring, random_spectrum, parent_filter_tolerance):
    """Peaks close to the parent mass should be filtered out"""
    
    parent, data = random_spectrum(scoring)
    
    # Make sure we have *m/z* in the parent+/-parent_filter_tolerance range
    data = data.copy()
    data[0, MZ] = parent + np.random.random() * parent_filter_tolerance
    # Make sure that excluding range is strict
    data[1, MZ] = parent + parent_filter_tolerance
    data[2, MZ] = parent - parent_filter_tolerance
    
    filtered = np.sort(filter_data(parent, data, 0, parent_filter_tolerance,
                                   50, 6, 50,
                                   scoring=='cosine', 'dot' if scoring=='cosine' else 'sum'), axis=0)
                                   
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
def test_filter_data_min_intensity(scoring, random_spectrum, min_intensity):
    """Peaks higher than `min_intensity` % of maximum intensity should be
       filtered out"""
    
    parent, data = random_spectrum(scoring)
    
    filtered = np.sort(filter_data(parent, data, min_intensity, 0,
                                   0, 0, 50, scoring=='cosine', 'dot' if scoring=='cosine' else 'sum'), axis=0)
    
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
def test_filter_data_window(scoring, random_spectrum, matched_peaks_window,
                            min_matched_peaks_search):
    """If `matched_peaks_window` or `min_matched_peaks_search` is zero,
       no peaks should be filtered.
    """
    
    parent, data = random_spectrum(scoring)
    
    filtered = np.sort(filter_data(parent, data, 0, 0, matched_peaks_window,
                                   min_matched_peaks_search,
                                   scoring=='cosine', 'dot' if scoring=='cosine' else 'sum'), axis=0)
    
    if matched_peaks_window == 0 or min_matched_peaks_search == 0:
        assert filtered.shape == data.shape
        assert filtered[:, MZ] == pytest.approx(np.sort(data[:, MZ], axis=0))
    else:
        assert filtered.shape <= data.shape
            

def test_filter_data_non_contiguous(scoring, random_spectrum, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search, mz_min):
    """f-contiguous arrays should give same results than c-contiguous arrays.
    """
    
    parent, data = random_spectrum(scoring)
    
    filtered = np.sort(filter_data(parent, data, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search, mz_min, scoring=='cosine', 'dot' if scoring=='cosine' else 'sum'), axis=0)
    data = np.asfortranarray(data, dtype=data.dtype)
    filtered_nc = np.sort(filter_data(parent, data, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search, mz_min, scoring=='cosine', 'dot' if scoring=='cosine' else 'sum'), axis=0)
    
    assert filtered == pytest.approx(filtered_nc)
    

def test_filter_data_reversed(scoring, random_spectrum, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search, mz_min):
    """Reversed arrays should give same results than non-reversed arrays.
    """
    
    parent, data = random_spectrum(scoring)
    
    filtered = np.sort(filter_data(parent, data, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search, mz_min, scoring=='cosine', 'dot' if scoring=='cosine' else 'sum'), axis=0)
    filtered_r = np.sort(filter_data(parent, data[::-1], min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search, mz_min, scoring=='cosine', 'dot' if scoring=='cosine' else 'sum'), axis=0)
    
    assert filtered == pytest.approx(filtered_r)
    

def test_filter_data_empty(scoring, min_intensity, parent_filter_tolerance,
                           matched_peaks_window, min_matched_peaks_search,
                           mz_min):
    
    parent = 152.569
    data = np.empty((0, 2), dtype=np.float32)
    
    filtered = filter_data(parent, data, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search, mz_min, scoring=='cosine', 'dot' if scoring=='cosine' else 'sum')
    assert filtered.size == 0

@pytest.mark.python
@pytest.mark.skipif(not IS_CYTHONIZED, reason="libmetgem should be cythonized")
def test_filter_data_python_cython(scoring, random_spectrum, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search, mz_min):
    """Cythonized `filter_data` and it's fallback Python version should give the
       same results.
    """
    
    parent, data = random_spectrum(scoring)
    
    filtered_p = np.sort(filter_data.__wrapped__(parent, data, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search, mz_min, scoring=='cosine', 'dot' if scoring=='cosine' else 'sum'), axis=0)
    filtered_c = np.sort(filter_data(parent, data, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search, mz_min, scoring=='cosine', 'dot' if scoring=='cosine' else 'sum'), axis=0)
    
    assert filtered_p.shape == filtered_c.shape
    assert filtered_p == pytest.approx(filtered_c)
