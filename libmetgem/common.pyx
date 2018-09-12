# cython: language_level=3
# distutils: language=c++

cimport numpy as np
cimport cython
from cython.view cimport array as cvarray
from libcpp.vector cimport vector
from cython.parallel cimport prange

DEF MZ = 0
DEF INTENSITY = 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float[:,:] arr_from_vector(vector[peak_t] v):
    cdef:
        int i
        float[:,:] a = cvarray(shape=(v.size(), 2), itemsize=sizeof(float), format='f')
        
    for i in prange(<int>v.size(), nogil=True):
        a[i, MZ] = v[i].mz
        a[i, INTENSITY] = v[i].intensity
        
    return a
   
@cython.boundscheck(False)
@cython.wraparound(False)
cdef peak_t *np_arr_pointer(np.ndarray[np.float32_t, ndim=2] data):
    if data.strides[0] < 0:
        return <peak_t *>&data[data.shape[0] - 1, 0]
    else:
        return <peak_t *>&data[0, 0]