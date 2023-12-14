"""
Test `libmetgem.network.generate_network`
"""

import pytest
import numpy as np
from scipy.sparse import csr_matrix

from libmetgem import IS_CYTHONIZED
from libmetgem.network import generate_network
from funcs import generate_network_f

from data import (matrix, random_matrix,
                  pairs_min_cosine, top_k)


def test_random_matrix(random_matrix):
    """Make sure that random_matrix is 2D, square, symmetric, with diagonal full
       of 1, with values between 0 and 1 and with dtype np.float32"""
       
    _, matrix = random_matrix
       
    assert len(matrix.shape) == 2
    assert matrix.shape[0] == matrix.shape[1]
    assert matrix == pytest.approx(matrix.T)
    diag = np.diag(matrix)
    assert np.count_nonzero(diag-1) == 0
    assert matrix.max() == 1
    assert matrix.min() >= 0
    assert matrix.dtype == np.float32
    
    
@pytest.mark.parametrize("sparse", [True, False])
def test_generate_network_empty(generate_network_f, sparse):
    """An empty matrix should not throw an error but return an empty array.
    """

    matrix = np.empty((0,0), dtype=np.float32)
    if sparse:
        matrix = csr_matrix(matrix)
    interactions = generate_network_f(matrix, [], 0.65, 10)
    
    assert interactions.size == 0
    
    
@pytest.mark.parametrize("sparse", [True, False])
def test_generate_network_matrix_larger_than_list(random_matrix, generate_network_f, sparse):
    """
        If matrix is larger than list, only part of the matrix will be used but
        no error should be thrown.
    """
    
    mzs, matrix = random_matrix

    mzs = mzs[:-2]
    max_index = matrix.shape[0]
    
    if sparse:
        matrix = csr_matrix(matrix)
    
    interactions = generate_network_f(matrix, mzs, 0, 10)
    
    assert max_index-1 not in interactions['Source']
    assert max_index-1 not in interactions['Target']
    assert max_index-2 not in interactions['Source']
    assert max_index-2 not in interactions['Target']
    

@pytest.mark.parametrize("sparse", [True, False])    
def test_generate_network_list_larger_than_matrix(random_matrix, generate_network_f, sparse):
    """
        If list is larger than matrix, only part of the list will be used but
        no error should be thrown.
    """
    
    mzs, matrix = random_matrix

    mzs = mzs + [1200.14225, 258.4475]
    max_index = len(mzs)
    
    if sparse:
        matrix = csr_matrix(matrix)
    
    interactions = generate_network_f(matrix, mzs, 0, 10)
    
    assert max_index-1 not in interactions['Source']
    assert max_index-1 not in interactions['Target']
    assert max_index-2 not in interactions['Source']
    assert max_index-2 not in interactions['Target']
    
    
@pytest.mark.parametrize("sparse", [False, True, False])
def test_generate_network_all_zero(random_matrix, generate_network_f, sparse):
    """
        If all filtering parameters are set to zero, we should get all possibles
        interactions excluding self loops.
    """

    mzs, matrix = random_matrix
    
    max_size = np.count_nonzero(np.triu(matrix)) - matrix.shape[0]
    
    if sparse:
        matrix = csr_matrix(matrix)
    
    interactions = generate_network_f(matrix, mzs, 0, 0)
    
    assert interactions.shape[0] == max_size
    

@pytest.mark.parametrize("sparse", [True, False])
def test_generate_network_high_top_k(random_matrix, generate_network_f, sparse):
    """
        If top_k is high and pairs_min_cosine is zero, we should get all
        possibles interactions excluding self loops.
    """
    
    mzs, matrix = random_matrix

    max_size = np.count_nonzero(np.triu(matrix)) - matrix.shape[0]
    top_k = matrix.shape[0]
    
    if sparse:
        matrix = csr_matrix(matrix)
    
    interactions = generate_network_f(matrix, mzs, 0, top_k)
    
    assert interactions.shape[0] == max_size
    

@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("pairs_min_cosine", [1, 2, 3])
def test_generate_network_high_pairs_min_cosine(random_matrix, generate_network_f,
                                                pairs_min_cosine, top_k, sparse):
    """
        If pairs_min_cosine is higher than 1, we should get an empty array.
    """
    
    mzs, matrix = random_matrix

    if sparse:
        matrix = csr_matrix(matrix)

    interactions = generate_network_f(matrix, mzs,
                                    pairs_min_cosine, top_k)
    
    assert interactions.size == 0
    

@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("pairs_min_cosine", [0, 0.3, 0.7, 1])
def test_generate_network_self_loop(random_matrix, generate_network_f,
                                    pairs_min_cosine, top_k, sparse):
    """
        Output array should not include self-loops
    """
    
    mzs, matrix = random_matrix

    if sparse:
        matrix = csr_matrix(matrix)

    interactions = generate_network_f(matrix, mzs,
                                    pairs_min_cosine, top_k)
    count = len([source for source, target, _, _ in interactions if source == target])
    
    assert count == 0
    

@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("pairs_min_cosine", [-0.2, 0, 0.3, 0.7])    
def test_generate_network_pairs_min_cosine(random_matrix, generate_network_f,
                                           pairs_min_cosine, sparse):
    """
        All cosine scores in outputted array should be strictly higher than
        pairs_min_cosine. Values lower than pairs_min_cosine and negative cosine
        scores should be filtered out.
        
    """
    
    mzs, matrix = random_matrix
    
    matrix = matrix.copy()
    
    matrix[0, 1] = matrix[1, 0] = pairs_min_cosine + 0.1
    matrix[0, 2] = matrix[2, 0] = pairs_min_cosine - 0.1

    if sparse:
        matrix = csr_matrix(matrix)

    interactions = generate_network_f(matrix, mzs,
                                    pairs_min_cosine, 0)
                                    
    seen1, seen2 = False, False
    for source, target, delta, cosine in interactions:
        assert cosine > pairs_min_cosine
        assert matrix[source, target] == cosine
        if (source == 0 and target == 1) or (source == 1 and target == 0):
            seen1 = True
        if (source == 0 and target == 2) or (source == 2 and target == 0):
            seen2 = True
            
    if matrix[0, 1] >= 0:
        assert seen1
    assert not seen2
    
    
@pytest.mark.slow
@pytest.mark.python
@pytest.mark.skipif(not IS_CYTHONIZED, reason="libmetgem should be cythonized")
@pytest.mark.parametrize("sparse", [True, False])
def test_generate_network_python_cython(random_matrix,
                              pairs_min_cosine, top_k, sparse):
    """Cythonized `generate_network` and it's fallback Python version should
       give the same results.
    """
    
    mzs, matrix = random_matrix
    
    if sparse:
        matrix = csr_matrix(matrix)
    
    interactions_p = generate_network.__wrapped__(matrix, mzs,
                                    pairs_min_cosine, 0)
    interactions_c = generate_network(matrix, mzs,
                                    pairs_min_cosine, 0)
    
    assert interactions_p.shape == interactions_c.shape
    assert interactions_p.dtype == interactions_c.dtype
    
    for name, (dtype, _) in interactions_c.dtype.fields.items():  
        if np.issubdtype(dtype.type, np.floating):
            assert pytest.approx(interactions_p[name]) == interactions_c[name]
        else:
            assert np.array_equal(interactions_p[name], interactions_c[name])


def test_generate_network_sparse_dense(random_matrix,
                              pairs_min_cosine, top_k,):
    """`generate_network` should output the same result with sparse or dense input
    if input is the same"""
    
    mzs, matrix = random_matrix
    
    interactions_s = generate_network(csr_matrix(matrix), mzs,
                                    pairs_min_cosine, 0)
    interactions_d = generate_network(matrix, mzs,
                                    pairs_min_cosine, 0)
    
    assert interactions_s.shape == interactions_d.shape
    assert interactions_s.dtype == interactions_d.dtype
    
    for name, (dtype, _) in interactions_d.dtype.fields.items():  
        if np.issubdtype(dtype.type, np.floating):
            assert pytest.approx(interactions_s[name]) == interactions_d[name]
        else:
            assert np.array_equal(interactions_s[name], interactions_d[name])