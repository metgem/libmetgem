# cython: language_level=3
# distutils: language=c++

import errno
import os

cimport cython
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libc.stdlib cimport strtof as std_strtof, strtol
from libc.string cimport strncmp, strncpy, strcpy, strcspn, strlen
from libc.stdio cimport fopen, fclose, fgets, FILE

from ._common cimport peak_t, arr_from_peaks_vector

cdef enum:
    MAX_KEY_SIZE = 64
    MAX_VALUE_SIZE = 2048
    MAX_LINE_SIZE = 2051 # 2048 characters + '\r\n' + '\0'

cdef extern from "<string.h>" nogil:
    char *strchr (char *string, int c)
    
    
# Implement strlwr as it is only available on windows
cdef extern from "<ctype.h>" nogil:
    int tolower(int c)

cdef char* strlwr(char* string) nogil:
    cdef int i=0

    while string[i] != b'\0':
        string[i] = tolower(string[i])
        i += 1

    return string

cdef extern from *:
    '''
    #ifdef WIN32
        #define CHARSET "mbcs"
    #else
        #define CHARSET "UTF-8"
    #endif
    '''
    extern const char* CHARSET
    
    
cdef inline double strtof(char* string, char **endptr) nogil:
    cdef char *ptr = NULL
    
    # Allow comma as decimal separator
    ptr = strchr(string, b',')
    if ptr > string:
        string[ptr-string] = b'.'
    return std_strtof(string, endptr)
    
    
cdef void read_data(char line[MAX_LINE_SIZE], vector[peak_t] *peaklist, FILE *fp) nogil:
    """Read peak list from file.
       First line has already been read and has to be passed as first argument.
       peak list is read into `peaklist`, which is cleared as first step."""
       
    cdef:
        peak_t peak
        float value
        char *ptr = NULL
        
    peaklist.clear()
    while True:
        if strncmp(line, 'END IONS', 8) == 0:
            return
        else:
            value = std_strtof(line, &ptr)
            if value > 0:
                peak.mz = value
                peak.intensity = std_strtof(ptr, NULL)
                peaklist.push_back(peak)
                
        if fgets(line, MAX_LINE_SIZE, fp) == NULL:
            return
           
           
cdef tuple read_entry(FILE * fp, bint ignore_unknown=False):
    """Read a spectrum entry (params and peaklist) from file
    """
    
    cdef:
        dict params = {}
        char *ptr = NULL
        char line[MAX_LINE_SIZE]
        vector[peak_t] peaklist
        int charge
        size_t pos
        char key[MAX_KEY_SIZE]
        char value[MAX_VALUE_SIZE]
        
    
    while fgets(line, MAX_LINE_SIZE, fp) != NULL:
        # Ignore blank lines
        if line[0] == b'\n' or line[0] == b'\r':
            continue
            
        ptr = strchr(line, b'=')
        if ptr > line:
            if strncmp(line, 'PEPMASS', 7) == 0:
                params['pepmass'] = strtof(line+8, NULL)
            elif strncmp(line, 'FEATURE_ID', 10) == 0:
                params['feature_id'] = strtol(line+11, &ptr, 10)
            elif strncmp(line, 'CHARGE', 6) == 0:
                charge = strtol(line+7, &ptr, 10)
                if strncmp(ptr, '-', 1) == 0:
                    charge *= -1
                params['charge'] = charge
            elif strncmp(line, 'RTINSECONDS', 11) == 0:
                params['rtinsecond'] = strtof(line+12, NULL)
            elif strncmp(line, 'MSLEVEL', 7) == 0:
                params['mslevel'] = strtol(line+8, &ptr, 10)
            elif not ignore_unknown:
                pos = ptr - line
                strncpy(key, line, pos)
                key[pos] = b'\0'
                key = strlwr(key)
                strcpy(value, line+pos+1)
                if strlen(value) > 0:
                    pos = strcspn(value, '\r\n')
                    value[pos] = b'\0'
                    params[key.decode('UTF-8', 'ignore')] = value.decode('UTF-8', 'ignore')
        else:
            if not params or not 'pepmass' in params:
                # If no pepmass found, skip all ions
                while True:
                    if strncmp(line, 'END IONS', 8) == 0:
                        break
                    if fgets(line, MAX_LINE_SIZE, fp) == NULL:
                        break
                return params, np.empty((0, 2), dtype=np.float32)
            else:
                read_data(line, &peaklist, fp)
                if peaklist.size() > 0:
                    return params, arr_from_peaks_vector(peaklist)
                else:
                    return params, np.empty((0, 2), dtype=np.float32)
                

def read(str filename, bint ignore_unknown=False):
    cdef:
        bytes fname_bytes
        char *fname
        tuple entry
        char line[MAX_LINE_SIZE]
        FILE *fp

    fname_bytes = filename.encode(CHARSET)
        
    fname = fname_bytes
        
    fp = fopen(fname, 'r')
    if fp == NULL:
        raise FileNotFoundError(errno.ENOENT,
                                os.strerror(errno.ENOENT),
                                filename)

    while fgets(line, MAX_LINE_SIZE, fp) != NULL:
        if strncmp(line, 'BEGIN IONS', 10) == 0:
            entry = read_entry(fp, ignore_unknown)
            if entry:
                yield entry
            
    fclose(fp)