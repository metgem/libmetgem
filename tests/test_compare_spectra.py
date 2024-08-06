"""
Test `libmetgem.cosine.compare_spectra`
"""

import pytest
import numpy as np

from libmetgem import IS_CYTHONIZED
from libmetgem.cosine import compare_spectra
from funcs import compare_spectra_f

from data import (known_spectra_comparisons, random_spectrum, another_random_spectrum,
                  mz_tolerance, min_matched_peaks)
 
DTYPE = np.dtype([('ix1', '<u2'), ('ix2', '<u2'), ('score', '<f8'), ('type', '<u1')])

def test_compare_spectra_known(known_spectra_comparisons, compare_spectra_f):
    """`compare_spectra` should return `expected` values"""
    
    scoring, mz1, mz2, data1, data2, mz_tolerance, expected = known_spectra_comparisons
    expected = np.array(expected, dtype=DTYPE)
    
    comparison = compare_spectra_f(mz1, data1, mz2, data2, mz_tolerance, scoring)
    assert comparison.dtype == DTYPE
    assert comparison.size == len(expected)
    if comparison.size > 0:
        assert np.array_equal(comparison['ix1'], expected['ix1'])
        assert np.array_equal(comparison['ix2'], expected['ix2'])
        assert np.array_equal(comparison['type'], expected['type'])
        assert comparison['score'] == pytest.approx(expected['score'], abs=1e-4)


def test_compare_spectra_empty(scoring, random_spectrum, mz_tolerance, compare_spectra_f):
    """Empty spectra should result in an empty comparison array"""
    
    mz, _ = random_spectrum(scoring)
    
    data = np.empty((0, 2), dtype=np.float32)
    comparison = compare_spectra_f(mz, data,
                                   mz, data,
                                   mz_tolerance,
                                   scoring)
    assert comparison.size == 0


def test_compare_spectra_identity(scoring, random_spectrum, mz_tolerance, compare_spectra_f):
    """Comparison array between one spectra and itself should always include always include all peaks."""
    
    mz, data = random_spectrum(scoring)
    
    comparison = compare_spectra_f(mz, data, mz, data, mz_tolerance, scoring)
    assert comparison.size == data.shape[0]
    expected_sum = 1.0 if scoring == 'cosine' else 2.0
    assert comparison['score'].sum() == pytest.approx(expected_sum)
    

@pytest.mark.parametrize("random_spectrum", range(4), indirect=True)
@pytest.mark.parametrize("another_random_spectrum", range(4), indirect=True)
def test_compare_spectra_swapped(scoring, random_spectrum, another_random_spectrum,
                                 mz_tolerance, min_matched_peaks, compare_spectra_f):
    """Comparisons should only differs by indices (swapped) if comparing
       spectrum 1 with spectrum 2 or spectrum 2 with spectrum 1."""
    
    mz1, data1 = random_spectrum(scoring)
    mz2, data2 = another_random_spectrum(scoring)
    
    comparison = compare_spectra_f(mz1, data1, mz2, data2, mz_tolerance, scoring)
    comparison_swapped = compare_spectra_f(mz2, data2, mz1, data1, mz_tolerance, scoring)
    assert comparison.size == comparison_swapped.size
    if comparison.size > 0:
        assert np.array_equal(comparison['ix1'], comparison_swapped['ix2'])
        assert np.array_equal(comparison['ix2'], comparison_swapped['ix1'])
        assert np.array_equal(comparison['score'], comparison_swapped['score'])
        assert np.array_equal(comparison['type'], comparison_swapped['type'])


@pytest.mark.python
@pytest.mark.skipif(not IS_CYTHONIZED, reason="libmetgem should be cythonized")
@pytest.mark.parametrize("random_spectrum", range(4), indirect=True)
@pytest.mark.parametrize("another_random_spectrum", range(4), indirect=True)
def test_compare_spectra_python_cython(scoring, random_spectrum, another_random_spectrum,
                                       mz_tolerance, min_matched_peaks):
    """Cythonized `compare_spectra` and it's fallback Python version should give
        the same results.
    """
    
    args = (*random_spectrum(scoring), *another_random_spectrum(scoring), mz_tolerance)
    comparison_p = compare_spectra.__wrapped__(*args, scoring)
    comparison_c = compare_spectra(*args, scoring)
    
    assert comparison_p.size == comparison_c.size
    if comparison_p.size > 0:
        assert np.array_equal(comparison_p['ix1'], comparison_c['ix1'])
        assert np.array_equal(comparison_p['ix2'], comparison_c['ix2'])
        assert np.array_equal(comparison_p['type'], comparison_c['type'])
        assert comparison_p['score'] == pytest.approx(comparison_c['score'])
    
