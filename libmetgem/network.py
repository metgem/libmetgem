"""
    Generate network from distance matrices.
"""

from ._loader import load_cython

from typing import List, Callable

import numpy as np

@load_cython
def generate_network(scores_matrix: np.ndarray, mzs: List[float],
                     pairs_min_cosine: float, top_k: float,
                     callback: Callable[[int], bool]=None) -> np.ndarray:
    """
        Generate a network graph from a pairwise distance matrix applying a
        `TopK`_ algorithm.
    
    Args:
        scores_matrix: pairwise distance matrix.
        mzs: List of parent ions' *m/z*.
        pairs_min_cosine: minimum cosine score between a pair of spectra in
            order for an edge to be kept in the network.
        top_k: edges between two nodes are kept only if both nodes are within
            each other's `top_k` most similar nodes.
        callback: function called to track progress of computation. First
            parameter (`int`) is the number of spectra computed since last call.
            It should return True if processing should continue, or False if
            computations should stop.
            
    Returns:
        2D structured array with 4 fields:
        * Source (int): index of source node.
        * Target (int): index of target node.
        * Delta MZ (np.float32): difference between *m/z* of source parent and
          *m/z* of target parent.
        * Cosine (np.float32): cosine score between source and target nodes.
    
    See Also:
        compute_distance_matrix, cosine_score
        
    .. _TopK:
        https://bix-lab.ucsd.edu/display/Public/Molecular+Networking+Documentation#MolecularNetworkingDocumentation-AdvancedNetworkOptions
        
    """
    
    interactions = []
    size = min(scores_matrix.shape[0], len(mzs))

    triu = np.triu(scores_matrix)
    triu[triu <= max(0, pairs_min_cosine)] = 0
    for i in range(size):
        # indexes = np.argpartition(triu[i,], -top_k)[-top_k:] # Should be faster and give the same results
        indexes = np.argsort(triu[i,])
        if top_k > 0:
            indexes = indexes[-top_k:]
        indexes = indexes[triu[i, indexes] > 0]

        for index in indexes:
            interactions.append((i, index, mzs[i] - mzs[index], triu[i, index]))

        if callback is not None and i > 0 and i % 100 == 0:
            if not callback(100):
                return []

    interactions = np.array(interactions, dtype=[('Source', int), ('Target', int), ('Delta MZ', np.float32),
                                                 ('Cosine', np.float32)])

    size = interactions.shape[0]
    if callback is not None and size % 100 != 0:
        if not callback(size % 100):
            return []
        callback(-size)  # Negative value means new maximum

    interactions = interactions[np.argsort(interactions, order='Cosine')[::-1]]

    # Top K algorithm, keep only edges between two nodes if and only if each of the node appeared in each other’s respective top k most similar nodes
    mask = np.zeros(interactions.shape[0], dtype=bool)
    for i, (x, y, _, _) in enumerate(interactions):
        x_ind = np.where(np.logical_or(interactions['Source'] == x, interactions['Target'] == x))[0]
        if top_k > 0:
            x_ind = x_ind[:top_k]
        y_ind = np.where(np.logical_or(interactions['Source'] == y, interactions['Target'] == y))[0]
        if top_k > 0:
            y_ind = y_ind[:top_k]
        if (x in interactions[y_ind]['Source'] or x in interactions[y_ind]['Target']) \
                and (y in interactions[x_ind]['Source'] or y in interactions[x_ind]['Target']):
            mask[i] = True
        if callback is not None and i > 0 and i % 100 == 0:
            if not callback(100):
                return []
    interactions = interactions[mask]

    if callback is not None and size % 100 != 0:
        callback(size % 100)

    return interactions