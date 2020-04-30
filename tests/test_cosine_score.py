"""
Test `libmetgem.cosine.cosine_score`
"""

import pytest
import numpy as np

from libmetgem import IS_CYTHONIZED
from libmetgem.cosine import cosine_score
from funcs import cosine_score_f

from data import (known_cosines, random_spectrum, another_random_spectrum,
                  mz_tolerance, min_matched_peaks)
 

def test_cosine_score_known(known_cosines, cosine_score_f):
    """`cosine_score` should return `expected` values"""
    
    mz1, mz2, data1, data2, mz_tolerance, \
        min_matched_peaks, expected = known_cosines
    
    cosine = cosine_score_f(mz1, data1, mz2, data2,
                          mz_tolerance, min_matched_peaks)
    assert pytest.approx(cosine, rel=1e-4) == expected


def test_cosine_score_empty(random_spectrum, mz_tolerance, min_matched_peaks, cosine_score_f):
    """Empty spectra should result in a null cosine score"""
    
    mz, _ = random_spectrum
    
    data = np.empty((0, 2), dtype=np.float32)
    cosine = cosine_score_f(mz, data,
                          mz, data,
                          mz_tolerance, min_matched_peaks)
    assert pytest.approx(cosine) == 0.0


def test_cosine_score_identity(random_spectrum, mz_tolerance, min_matched_peaks, cosine_score_f):
    """Cosine scores between one spectra and itself should always be 1."""
    
    mz, data = random_spectrum
    
    cosine = cosine_score_f(mz, data, mz, data,
                          mz_tolerance, min_matched_peaks)
    assert pytest.approx(cosine) == 1.0
    

@pytest.mark.parametrize("random_spectrum", range(4), indirect=True)
@pytest.mark.parametrize("another_random_spectrum", range(4), indirect=True)
def test_cosine_score_swapped(random_spectrum, another_random_spectrum,
                              mz_tolerance, min_matched_peaks, cosine_score_f):
    """Cosine scores should not change if comparing spectrum 1 with spectrum 2
        or spectrum 2 with spectrum 1."""
    
    mz1, data1 = random_spectrum
    mz2, data2 = another_random_spectrum
    
    cosine = cosine_score_f(mz1, data1, mz2, data2,
                          mz_tolerance, min_matched_peaks)
    cosine_swapped = cosine_score_f(mz2, data2, mz1, data1,
                          mz_tolerance, min_matched_peaks)
    assert pytest.approx(cosine) == cosine_swapped


@pytest.mark.python
@pytest.mark.skipif(not IS_CYTHONIZED, reason="libmetgem should be cythonized")
@pytest.mark.parametrize("random_spectrum", range(4), indirect=True)
@pytest.mark.parametrize("another_random_spectrum", range(4), indirect=True)
def test_cosine_score_python_cython(random_spectrum, another_random_spectrum,
                                    mz_tolerance, min_matched_peaks):
    """Cythonized `cosine_score` and it's fallback Python version should give
        the same results.
    """
    
    args = (*random_spectrum, *another_random_spectrum,
            mz_tolerance, min_matched_peaks)
    cosine_p = cosine_score.__wrapped__(*args)
    cosine_c = cosine_score(*args)
    assert pytest.approx(cosine_p) == cosine_c
    
