import numpy as np
cimport numpy as np
import cython
from libcpp.vector cimport vector
from libcpp.set cimport set as unordered_set
from libc.math cimport sqrt

cdef double ward(int size_a, int size_b, const vector[double]& pos_a, const vector[double]& pos_b, int dim) :
    """calculates the ward for one cluster to another"""
    cdef double diff, result = 0.0
    cdef double s = (size_a * size_b) / (size_a + size_b)
    for i in range(dim):
        diff = pos_a[i] - pos_b[i]
        result += diff * diff
    return s * result

cdef vector[int] get_top_k(int i, vector[int]& size, vector[vector[double]]& pos, unordered_set[int]& active, int k, int dim):
    """Finds the nearest k neighbours"""
    cdef vector[int] active_, top_k
    cdef vector[double] dists
    cdef double ds
    cdef int a = active.size() - 1
    cdef int size_i = size[i]
    cdef vector[double] pos_i = pos[i]
    cdef int p = min(k, a)
    
    top_k.clear()
    top_k.reserve(p)
    active_.reserve(a)
    dists.reserve(a)
    
    for j in active:
        if j != i:
            active_.push_back(j)
            ds = ward(size_i, size[j], pos_i, pos[j], dim)
            dists.push_back(ds)

    sorting = np.argsort(dists)[:k]
    
    for index in range(p):
        top_k.push_back(active_[sorting[index]])
    
    return top_k
   

cpdef knn_chain(np.ndarray[np.double_t, ndim=2] X, int k = 5):
    """Calculates the NN chain algorithm with on the fly distances"""

    cdef vector[vector[double]] dendrogram, pos = X
    cdef int i, j, m, index, new_index, tmp_size
    cdef int n = pos.size(), dim = pos[0].size()
    cdef double tmp_dist
    cdef vector[int] size, chain, top_k
    cdef vector[double] dists, centroid
    cdef vector[vector[int]] knn
    active = {i for i in range(n)}
    mapping = {i: i for i in range(n)}
    reverse_mapping = {i: {i} for i in range(n)}

    pos.reserve(2 * n - 3)
    top_k.reserve(k)
    dendrogram.reserve(2 * n - 1)
    size.reserve(2 * n - 1)
    centroid.reserve(dim)
    for i in range(n):
        size.push_back(1)

    # Activate loop
    while active:

        if len(active) == 2:
            i, j = active
            size_ = size[i] + size[j]
            dist_ = ward(size[i], size[j], pos[i], pos[j], dim)
            dendrogram.push_back([i, j, np.sqrt(2 * dist_), size_])
            return dendrogram
        
        # New chain
        if not len(chain):
            i = next(iter(active))
            chain.push_back(i)

            top_k = get_top_k(i, size, pos, active, k, dim)
            knn.push_back(top_k)

        while len(chain):

            i = chain.back()
            m = -1
            for index in range(knn.back().size()):
                if knn.back()[index] in active:
                    m = index
                    break
            if m <= 0:
                if m < 0:
                    knn.pop_back()
                    knn.push_back(get_top_k(i, size, pos, active, k, dim))
                j = knn.back()[0]
            else:
                indices = set()
                for index in range(m):
                    indices |= reverse_mapping[knn.back()[index]]
                    
                clusters = set()
                for index in indices:
                    clusters.add(mapping[index])
                    
                top_k = list(clusters) + [knn.back()[m]]
                dists = [ward(size[i], size[j], pos[i], pos[j], dim) for j in top_k]
                j = top_k[np.argmin(dists)]

            if chain.size() > 1 and chain[chain.size()-2] == j:
                break
            
            chain.push_back(j)
            top_k = get_top_k(j, size, pos, active, k, dim)
            knn.push_back(top_k)

        # Merge
        dist_ = ward(size[i], size[j], pos[i], pos[j], dim)
        size_ = size[i] + size[j]
        dendrogram.push_back([i, j, np.sqrt(2 * dist_), size_])
    
        # Update variables
        centroid = vector[double](dim)
        for index in range(dim):
            centroid[index] = (size[i] * pos[i][index] + size[j] * pos[j][index]) / size_
        
        pos.push_back(centroid)
        
        new_index = len(size)
        size.push_back(size_)
        
        # Update mapping
        for index in reverse_mapping[i] | reverse_mapping[j]:
            mapping[index] = new_index
            
        reverse_mapping[new_index] = reverse_mapping[i] | reverse_mapping[j]
        
        # Update active set
        active.remove(i)
        active.remove(j)
        active.add(new_index)

        chain = chain[:-2]
        knn = knn[:-2]

    return dendrogram