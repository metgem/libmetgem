"""
Test `libmetgem.neighbors.n_neighbors_from_perplexity`.
"""

import pytest
import numpy as np

from libmetgem.neighbors import n_neighbors_from_perplexity
    
    
@pytest.mark.parametrize('n_samples', [0, 15, 20, 30])
@pytest.mark.parametrize('perplexity', [0, 5, 10, 50])
def test_n_neighbors_from_perplexity(n_samples, perplexity):
    """`n_neighbors_from_perplexity` should always return 2D matrices.
    """
    
    n_neighbors = n_neighbors_from_perplexity(n_samples, perplexity)
    
    assert n_neighbors <= n_samples - 1
    assert n_neighbors <= 3. * perplexity + 1