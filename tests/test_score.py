"""
Test `libmetgem.cosine.cosine_score`
"""

import pytest
import numpy as np

from libmetgem import IS_CYTHONIZED
from libmetgem.cosine import cosine_score, entropy_score, weighted_entropy_score
from funcs import score_f_gen

from data import (known_scores, random_spectrum, another_random_spectrum,
                  mz_tolerance, min_matched_peaks)

def test_score_known(known_scores, score_f_gen):
    """`cosine_score`, `entropy_score` and `weighted_entropy_score` should return `expected` values"""
    
    scoring, mz1, mz2, data1, data2, mz_tolerance, \
        min_matched_peaks, expected = known_scores
        
    score_f = score_f_gen(scoring)
    score = score_f(mz1, data1, mz2, data2,
                    mz_tolerance, min_matched_peaks)
    assert pytest.approx(score, rel=1e-4) == expected


def test_score_empty(scoring, random_spectrum, mz_tolerance, min_matched_peaks, score_f_gen):
    """Empty spectra should result in a null score"""
    
    mz, _ = random_spectrum(scoring)
    
    data = np.empty((0, 2), dtype=np.float32)
    score_f = score_f_gen(scoring)
    score = score_f(mz, data.copy(),
                     mz, data.copy(),
                     mz_tolerance, min_matched_peaks)
    assert pytest.approx(score) == 0.0


def test_score_identity(scoring, random_spectrum, mz_tolerance, min_matched_peaks, score_f_gen):
    """Scores between one spectra and itself should always be 1."""
    
    mz, data = random_spectrum(scoring)
   
    score_f = score_f_gen(scoring)
    score = score_f(mz, data, mz, data,
                    mz_tolerance, min_matched_peaks)
    if scoring != 'cosine':
        assert pytest.approx(data[:, 1].sum()) == 1.0
        print(mz, data, score_f)
    assert pytest.approx(score) == 1.0
    

@pytest.mark.parametrize("random_spectrum", range(4), indirect=True)
@pytest.mark.parametrize("another_random_spectrum", range(4), indirect=True)
def test_score_swapped(scoring, random_spectrum, another_random_spectrum,
                              mz_tolerance, min_matched_peaks, score_f_gen):
    """Cosine scores should not change if comparing spectrum 1 with spectrum 2
        or spectrum 2 with spectrum 1."""
    
    mz1, data1 = random_spectrum(scoring)
    mz2, data2 = another_random_spectrum(scoring)
    
    score_f = score_f_gen(scoring)
    score = score_f(mz1, data1.copy(), mz2, data2.copy(),
                          mz_tolerance, min_matched_peaks)
    score_swapped = score_f(mz2, data2.copy(), mz1, data1.copy(),
                          mz_tolerance, min_matched_peaks)
    assert pytest.approx(score) == score_swapped


@pytest.mark.python
@pytest.mark.skipif(not IS_CYTHONIZED, reason="libmetgem should be cythonized")
@pytest.mark.parametrize("random_spectrum", range(4), indirect=True)
@pytest.mark.parametrize("another_random_spectrum", range(4), indirect=True)
def test_score_python_cython(scoring, random_spectrum, another_random_spectrum,
                             mz_tolerance, min_matched_peaks):
    """Cythonized `cosine_score` and it's fallback Python version should give
        the same results.
    """
    
    args = (*random_spectrum(scoring), *another_random_spectrum(scoring),
            mz_tolerance, min_matched_peaks)
    match scoring:
        case 'cosine':
            func = cosine_score
        case 'entropy':
            func = entropy_score
        case 'weighted_entropy':
            func = weighted_entropy_score
    score_p = func.__wrapped__(*args)
    score_c = func(*args)
    assert pytest.approx(score_p) == score_c
    
