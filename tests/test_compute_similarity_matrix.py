"""
Test `libmetgem.cosine.compute_similarity_matrix`.
"""

import pytest
import numpy as np

from libmetgem import IS_CYTHONIZED
from libmetgem.cosine import compute_similarity_matrix

from data import (matrix, random_spectra,
                  mz_tolerance, min_matched_peaks)
from funcs import compute_similarity_matrix_f


def test_matrix_shape(matrix):
    """`compute_similarity_matrix` should always return 2D matrices.
    """
    
    assert len(matrix.shape) == 2


def test_matrix_square(matrix):
    """`compute_similarity_matrix` should always return square matrices.
    """
    
    assert matrix.shape[0] == matrix.shape[1]


def test_matrix_diag(matrix):
    """`compute_similarity_matrix` should always return matrices with diagonal
        full of 1.
    """
        
    diag = np.diag(matrix)
    assert np.count_nonzero(diag-1) == 0


def test_matrix_symmetric(matrix):
    """`compute_similarity_matrix` should always return symmetric matrices.
    """
    
    assert matrix == pytest.approx(matrix.T)
    

def test_matrix_max(matrix):
    """`compute_similarity_matrix` returned matrices should not have values
        upper than 1.
    """
        
    assert matrix.max() == 1


def test_matrix_min(matrix):
    """`compute_similarity_matrix` returned matrices should not have values
        lower than 0.
    """
        
    assert matrix.min() >= 0
    
    
def test_matrix_dtype(matrix):
    """`compute_similarity_matrix` returned matrices should have dtype np.float32
    """
        
    assert matrix.dtype == np.float32
    
    
@pytest.mark.python
def test_matrix_sparse(scoring, random_spectra, mz_tolerance, min_matched_peaks):
    """Sparse and dense results should be equivalent.
    """
    
    mzs, spectra = random_spectra(scoring)
    
    matrix_s = compute_similarity_matrix(mzs, spectra,
                                         mz_tolerance,
                                         min_matched_peaks,
                                         scoring,
                                         dense_output=False)
    matrix = compute_similarity_matrix(mzs, spectra,
                                       mz_tolerance,
                                       min_matched_peaks,
                                       scoring)
    assert pytest.approx(matrix_s.toarray()) == matrix

    
@pytest.mark.python
@pytest.mark.skipif(not IS_CYTHONIZED, reason="libmetgem should be cythonized")
def test_matrix_python_cython(scoring, random_spectra, mz_tolerance, min_matched_peaks):
    """Cythonized `compute_similarity_matrix` and it's fallback Python version
        should give the same results.
    """
    
    mzs, spectra = random_spectra(scoring)
    
    matrix_p = compute_similarity_matrix.__wrapped__(mzs, spectra,
                                                   mz_tolerance,
                                                   min_matched_peaks,
                                                   scoring)
    matrix_c = compute_similarity_matrix(mzs, spectra,
                                       mz_tolerance,
                                       min_matched_peaks,
                                       scoring)
    assert pytest.approx(matrix_p) == matrix_c


def test_matrix_callback_count(scoring, random_spectra, mz_tolerance, min_matched_peaks, mocker, compute_similarity_matrix_f):
    """callback shoud be called one times per spectrum."""
        
    callback = mocker.Mock(return_value=True)
    
    mzs, spectra = random_spectra(scoring)
    matrix = compute_similarity_matrix_f(mzs, spectra, mz_tolerance, min_matched_peaks, scoring, callback=callback)
    
    assert callback.call_count == len(mzs)
    
def test_matrix_callback_abort(scoring, random_spectra, mz_tolerance, min_matched_peaks, mocker, compute_similarity_matrix_f):
    """process should be stopped if callback return False."""
        
    callback = mocker.Mock(return_value=False)
       
    mzs, spectra = random_spectra(scoring)
    matrix = compute_similarity_matrix_f(mzs, spectra, mz_tolerance, min_matched_peaks, scoring, callback=callback)
    
    if compute_similarity_matrix_f == 'python':
        assert callback.call_count == 1
    else:
        assert callback.call_count < len(mzs)
    
def test_matrix_callback_values(scoring, random_spectra, mz_tolerance, min_matched_peaks, mocker, compute_similarity_matrix_f):
    """callback shoud be called with increasing values. Cythonized one has no order."""
        
    callback = mocker.Mock(return_value=True)
       
    mzs, spectra = random_spectra(scoring)
    matrix = compute_similarity_matrix_f(mzs, spectra, mz_tolerance, min_matched_peaks, scoring, callback=callback)
    
    calls = [mocker.call(i) for i in range(len(mzs))]
    any_order = compute_similarity_matrix_f.variant == 'cython'
    callback.assert_has_calls(calls, any_order=any_order)
    
