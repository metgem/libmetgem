from ._loader import load_cython

from scipy.sparse import csr_matrix
import numpy as np

__all__ = ('kneighbors_graph_from_similarity_matrix', 'n_neighbors_from_perplexity')


def n_neighbors_from_perplexity(n_samples, perplexity):
    """
    Returns numbers of neighbors from number of samples and perplexity.
    Borrowed from Scikit-Learn's TNSE
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    """
    return min(n_samples - 1, int(3. * perplexity + 1))


@load_cython
def kneighbors_graph_from_similarity_matrix(matrix: csr_matrix, n_neighbors: int):
    """
    Transform `matrix` into a (weighted) graph of k nearest neighbors
    The transformed data is a sparse graph.
    `matrix` is assumed to be a sparse similarity matrix.
    This is equivalent to running KNeighborsTransformer((n_neighbors=n_neighbors, metric='precomputed')).fit_transform(1-matrix.toarray()) but with a lower memory consumption.
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsTransformer.html
    
    See `compute_similarity_matrix`
    """
    n_samples = matrix.shape[0]
    
    if n_neighbors > n_samples:
        raise ValueError("Expected n_neighbors <= n_samples,  but n_samples = {n_samples}, n_neighbors = {n_neighbors}")
    elif n_neighbors == n_samples:
        return matrix
        
    n_nonzero = n_samples * (n_neighbors+1)
    indptr = np.arange(0, n_nonzero + 1, n_neighbors+1)
    
    data = np.empty(n_nonzero, dtype=matrix.data.dtype)
    indices = np.empty(n_nonzero, dtype=int)
    row = np.ones(n_samples)

    for i in range(n_samples):
        indptr_range = slice(matrix.indptr[i], matrix.indptr[i+1])
        inds = matrix.indices[indptr_range]
        row[inds] = 1 - matrix.data[indptr_range]

        ind = np.argpartition(row, n_neighbors)
        ind = ind[:n_neighbors+1]
            
        indptr_range = slice(indptr[i], indptr[i+1])
        data[indptr_range] = row[ind]
        indices[indptr_range] = ind
        row[inds] = 1
    
    return csr_matrix((data, indices, indptr))
