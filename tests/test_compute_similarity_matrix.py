"""
Test `libmetgem.cosine.compute_similarity_matrix`.
"""

import pytest
import numpy as np

from libmetgem import IS_CYTHONIZED
from libmetgem.cosine import compute_similarity_matrix

from data import matrix, random_spectra, mz_tolerance, min_matched_peaks


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
