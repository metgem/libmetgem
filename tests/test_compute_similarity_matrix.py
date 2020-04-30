"""
Test `libmetgem.cosine.compute_similarity_matrix`.
"""

import pytest
import numpy as np

from libmetgem import IS_CYTHONIZED
from libmetgem.cosine import compute_similarity_matrix, compute_distance_matrix

from data import matrix, random_spectra, mz_tolerance, min_matched_peaks
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
@pytest.mark.skipif(not IS_CYTHONIZED, reason="libmetgem should be cythonized")
def test_matrix_python_cython(random_spectra, mz_tolerance, min_matched_peaks):
    """Cythonized `compute_similarity_matrix` and it's fallback Python version
        should give the same results.
    """
    
    mzs, spectra = random_spectra
    
    matrix_p = compute_similarity_matrix.__wrapped__(mzs, spectra,
                                                   mz_tolerance,
                                                   min_matched_peaks)
    matrix_c = compute_similarity_matrix(mzs, spectra,
                                       mz_tolerance,
                                       min_matched_peaks)
    assert pytest.approx(matrix_p) == matrix_c
    
    
def test_matrix_warnings(random_spectra, mz_tolerance, min_matched_peaks, compute_similarity_matrix_f):
    """`compute_distance_matrix` is deprecated. It should call `compute_similarity_matrix' with the same args"""
    
    mzs, spectra = random_spectra
    
    with pytest.deprecated_call():
        matrix1 = compute_distance_matrix(mzs, spectra, mz_tolerance, min_matched_peaks)
    matrix2 = compute_similarity_matrix_f(mzs, spectra, mz_tolerance, min_matched_peaks)
    assert pytest.approx(matrix1) == matrix2

def test_matrix_callback_count(random_spectra, mz_tolerance, min_matched_peaks, mocker, compute_similarity_matrix_f):
    """callback shoud be called one times per spectrum."""
        
    callback = mocker.Mock(return_value=True)
    
    mzs, spectra = random_spectra
    matrix = compute_similarity_matrix_f(mzs, spectra, mz_tolerance, min_matched_peaks, callback)
    
    assert callback.call_count == len(mzs)
    
def test_matrix_callback_abort(random_spectra, mz_tolerance, min_matched_peaks, mocker, compute_similarity_matrix_f):
    """process should be stopped if callback return False."""
        
    callback = mocker.Mock(return_value=False)
       
    mzs, spectra = random_spectra
    matrix = compute_similarity_matrix_f(mzs, spectra, mz_tolerance, min_matched_peaks, callback)
    
    if compute_similarity_matrix_f == 'python':
        assert callback.call_count == 1
    else:
        assert callback.call_count < len(mzs)
    
def test_matrix_callback_values(random_spectra, mz_tolerance, min_matched_peaks, mocker, compute_similarity_matrix_f):
    """callback shoud be called with increasing values. Cythonized one has no order."""
        
    callback = mocker.Mock(return_value=True)
       
    mzs, spectra = random_spectra
    matrix = compute_similarity_matrix_f(mzs, spectra, mz_tolerance, min_matched_peaks, callback)
    
    calls = [mocker.call(i) for i in range(len(mzs))]
    any_order = compute_similarity_matrix_f.variant == 'cython'
    callback.assert_has_calls(calls, any_order=any_order)
    
