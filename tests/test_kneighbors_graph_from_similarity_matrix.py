"""
Test `libmetgem.neighbors.kneighbors_graph_from_similarity_matrix`.
"""

import pytest
import numpy as np

from libmetgem import IS_CYTHONIZED
from libmetgem.neighbors import kneighbors_graph_from_similarity_matrix

from data import neighbors_graph, sparse_matrix, random_spectra, mz_tolerance, min_matched_peaks
from funcs import (compute_similarity_matrix_f,
                   kneighbors_graph_from_similarity_matrix_f)



def test_neighbors_shape(neighbors_graph):
    """`kneighbors_graph_from_similarity_matrix` should always return 2D matrices.
    """
    _, graph = neighbors_graph
    
    assert len(graph.shape) == 2


def test_neighbors_square(neighbors_graph):
    """`kneighbors_graph_from_similarity_matrix` should always return square matrices.
    """
    _, graph = neighbors_graph
    
    assert graph.shape[0] == graph.shape[1]

def test_neighbors_n_neighbors(neighbors_graph, request):
    """`kneighbors_graph_from_similarity_matrix` should always return matrices with
        no more than `n_neighbors` values higher than 0 per line.
    """
    
    n_neighbors, graph = neighbors_graph
        
    assert np.count_nonzero(graph.toarray(), axis=1).max() == n_neighbors
    

def test_neighbors_max(neighbors_graph):
    """`kneighbors_graph_from_similarity_matrix` returned matrices should not have values
        upper than 1.
    """
    n_neighbors, graph = neighbors_graph
        
    if n_neighbors == 0:
        assert graph.max() == 0
    else:
        assert graph.max() == 1


def test_neighbors_min(neighbors_graph):
    """`kneighbors_graph_from_similarity_matrix` returned matrices should not have values
        lower than 0.
    """
    _, graph = neighbors_graph
        
    assert graph.min() >= 0
    
    
def test_neighbors_dtype(neighbors_graph):
    """`kneighbors_graph_from_similarity_matrix` returned matrices should have dtype np.float32
    """
    _, graph = neighbors_graph
        
    assert graph.dtype == np.float32
    
    
@pytest.mark.parametrize('neighbors_graph', [0],
    indirect=True)
def test_neighbors_no_neighbors(neighbors_graph):
    """If n_neighbors = 0, `kneighbors_graph_from_similarity_matrix` should return a
    matrix full of 0.
    """
    _, graph = neighbors_graph
        
    assert np.count_nonzero(graph.toarray()) == 0
    
    
@pytest.mark.parametrize('n_neighbors', [15, 20])
def test_neighbors_too_much_neighbors(sparse_matrix, kneighbors_graph_from_similarity_matrix_f, n_neighbors):
    """If n_neighbors > matrix.shape[0], `kneighbors_graph_from_similarity_matrix` should raises a
    ValueError.
    """
    with pytest.raises(ValueError):
        kneighbors_graph_from_similarity_matrix_f(sparse_matrix, n_neighbors=n_neighbors)
        
        
def test_neighbors_identity(sparse_matrix, kneighbors_graph_from_similarity_matrix_f):
    """If n_neighbors > matrix.shape[0], `kneighbors_graph_from_similarity_matrix` should raises a
    ValueError.
    """
    m = kneighbors_graph_from_similarity_matrix_f(sparse_matrix, n_neighbors=sparse_matrix.shape[0]) 
    assert np.array_equal(m.toarray(), sparse_matrix.toarray())
    

def test_neighbors_check_values(sparse_matrix, neighbors_graph):
    """
    All non zeros values should be in `kneighbors_graph_from_similarity_matrix` output or replaced by 0.
    """
    
    _, graph = neighbors_graph

    for i, j in zip(*np.nonzero(sparse_matrix)):
        if graph[i,j] != 0.:
            assert graph[i,j] == pytest.approx(1. - sparse_matrix[i,j])

    
@pytest.mark.python
@pytest.mark.skipif(not IS_CYTHONIZED, reason="libmetgem should be cythonized")
def test_neighbors_python_cython(sparse_matrix):
    """Cythonized `kneighbors_graph_from_similarity_matrix` and it's fallback Python version
        should give equivalent results.
    """
    
    graph_p = kneighbors_graph_from_similarity_matrix.__wrapped__(sparse_matrix, n_neighbors=5)
    graph_c = kneighbors_graph_from_similarity_matrix(sparse_matrix, n_neighbors=5)
    
    assert np.allclose(graph_p.mean(axis=1), graph_c.mean(axis=1))
    assert np.array_equal(np.count_nonzero(graph_p.toarray(), axis=1),
                           np.count_nonzero(graph_c.toarray(), axis=1))
