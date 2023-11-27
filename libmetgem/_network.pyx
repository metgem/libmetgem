# cython: language_level=3
# distutils: language=c++

cimport cython
import numpy as np
cimport numpy as np
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.algorithm cimport (sort, partial_sort)
from libc.math cimport fabs

ctypedef struct interaction_t:
    int source
    int target
    double delta_mz
    double cosine
    
ctypedef struct element_t:
    int index
    double cosine
  
cdef bool compareElementsByCosine(const element_t &a, const element_t &b) noexcept nogil:
    return a.cosine > b.cosine
    
cdef bool compareInteractionsByCosine(const interaction_t &a, const interaction_t &b) noexcept nogil:
    if a.cosine > b.cosine:
        return True
    elif a.cosine < b.cosine:
        return False
    elif a.source > b.source:
        return True
    elif a.source < b.source:
        return False
    elif a.target > b.target:
        return True
    elif a.target < b.target:
        return False
    elif a.delta_mz > b.delta_mz:
        return True
    elif a.delta_mz < b.delta_mz:
        return False
    else:
        return False

@cython.boundscheck(False)
@cython.wraparound(False)
cdef vector[interaction_t] generate_network_nogil(const float[:,:] scores_matrix,
                                                  vector[double] mzvec,
                                                  double pairs_min_cosine,
                                                  size_t top_k,
                                                  object callback=None) noexcept nogil:
    cdef:
        vector[interaction_t] interactions, interactions2
        interaction_t inter
        size_t size = min(<size_t>scores_matrix.shape[0], mzvec.size())
        int i, j
        vector[element_t] row
        element_t element
        double cosine
        size_t length
        int x, y
        vector[int] x_ind, y_ind
        bool flag
        bool has_callback = callback is not None
      
    for i in range(size):
        row.clear()
        row.reserve(size-i)
        for j in range(i+1, size): # i+1 to remove self loops
            cosine = scores_matrix[i, j]
            if cosine > pairs_min_cosine >= 0:
                element.index = j
                element.cosine = cosine
                row.push_back(element)
                
        if top_k > 0:
            length = min(row.size(), top_k)
            partial_sort(row.begin(), row.begin() + length,
                         row.end(), &compareElementsByCosine)
        else:
            length = row.size()
            sort(row.begin(), row.end(), &compareElementsByCosine)
        
        for j in range(length):
            element = row[j]
            inter.source = i
            inter.target = element.index
            inter.delta_mz = mzvec[i]-mzvec[element.index]
            inter.cosine = element.cosine
            interactions.push_back(inter)
            
        if has_callback and i>0 and i % 100 == 0:
            with gil:
                if not callback(100):
                    interactions.clear()
                    return interactions
                    
    # Free memory
    row.clear()
    row.shrink_to_fit()
                
    size = interactions.size()
    if has_callback and size % 100 != 0:
        with gil:
            if not callback(size % 100):
                interactions.clear()
                return interactions
            callback(-size) # Negative value means new maximum
        
    sort(interactions.begin(), interactions.end(), &compareInteractionsByCosine)
        
    # Top K algorithm, keep only edges between two nodes if and only if each
    # of the node appeared in each other’s respective top k most similar nodes
    interactions2.reserve(size)
    for i in range(size):
        x = interactions[i].source
        y = interactions[i].target

        x_ind.clear()
        y_ind.clear()
        for j in range(size):
            if interactions[j].source == x or interactions[j].target == x:
                x_ind.push_back(j)
            if interactions[j].source == y or interactions[j].target == y:
                y_ind.push_back(j)

        if top_k > 0:
            x_ind.resize(min(top_k, x_ind.size()))
            y_ind.resize(min(top_k, y_ind.size()))
                
        flag = False
        for j in y_ind:
            if x == interactions[j].source or x == interactions[j].target:
                flag = True
        if flag:
            for j in x_ind:
                if y == interactions[j].source or y == interactions[j].target:
                    interactions2.push_back(interactions[i])
                    break
                    
        if has_callback and i>0 and i % 100 == 0:
            with gil:
                if not callback(100):
                    interactions2.clear()
                    return interactions
                
    if has_callback and size % 100 != 0:
        with gil:
            callback(size % 100)
            
    # Free memory
    interactions.clear()
    interactions.shrink_to_fit()
    x_ind.clear()
    x_ind.shrink_to_fit()
    y_ind.clear()
    y_ind.shrink_to_fit()
            
    return interactions2
    
    
def generate_network(const float[:,:] scores_matrix, vector[double] mzvec,
                     double pairs_min_cosine, size_t top_k, object callback=None):
    cdef:
        vector[interaction_t] interactions
        interaction_t inter
        int i
        np.dtype dt = np.dtype([('Source', int),
                                ('Target', int),
                                ('Delta MZ', np.float32),
                                ('Cosine', np.float32)])
        np.ndarray array
        
    interactions = generate_network_nogil(scores_matrix, mzvec,
                                          pairs_min_cosine, top_k,
                                          callback)
    array = np.empty((interactions.size(),), dtype=dt)
    for i in range(array.shape[0]):
        inter = interactions[i]
        array[i] = (inter.source, inter.target, inter.delta_mz, inter.cosine)
        
    return array
        
