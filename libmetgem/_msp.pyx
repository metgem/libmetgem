# cython: language_level=3
# distutils: language=c++

import errno
import os

cimport cython
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdlib cimport strtof, strtol
from libc.string cimport strncpy, strcpy, strcspn, strlen
from libc.stdio cimport fopen, fclose, fgets, FILE

from ._common cimport peak_t, arr_from_peaks_vector

cdef enum:
    MAX_KEY_SIZE = 64
    MAX_VALUE_SIZE = 2048
    MAX_LINE_SIZE = 2051 # 2048 characters + '\r\n' + '\0'

cdef extern from "<string.h>" nogil:
    '''
    #ifdef _WIN32
        #define strncasecmp(s1, s2, n)  _strnicmp(s1, s2, n)
    #endif
    '''
    int strncasecmp (const char *s1, const char *s2, size_t n)
    char *strchr (char *string, int c)
    
cdef extern from "<ctype.h>" nogil:
    int isspace(int character)

    
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
    #ifdef _WIN32
        #define CHARSET "mbcs"
    #else
        #define CHARSET "UTF-8"
    #endif
    '''
    extern const char* CHARSET
    
    
cdef inline bool isdelim(char s) noexcept nogil:
    return strchr(b'()[]{},;:', s) != NULL
    
    
cdef inline void trim(char* string) noexcept nogil:
    cdef:
        int i = 0
        int index = 0
        int last_char_index = 0
    
    while isspace(string[index]):
        index += 1
        
    if index > 0:
        while string[i+index] != b'\0':
            string[i] = string[i+index]
            if not isspace(string[i+index]):
                last_char_index = i
            i +=1
            
        if last_char_index:
            string[last_char_index+1] = b'\0'
        else:
            string[i] = b'\0'
    else:
        while string[i] != b'\0':
            if not isspace(string[i]):
                last_char_index = i
            i +=1
        if last_char_index:
            string[last_char_index+1] = b'\0'
    
    
cdef void read_data(char line[MAX_LINE_SIZE], vector[peak_t] *peaklist,
                    FILE *fp, int num_peaks) noexcept nogil:
    """Read peak list from file.
       First line has already been read and has to be passed as first argument.
       peak list is read into `peaklist`, which is cleared as first step."""
       
    cdef:
        peak_t peak
        float value
        char *ptr = NULL
        char *ptr2 = NULL
        int peaks_read = 0
        int i = 0
        
    peaklist.clear()
    while True:
        if line[0] == b'\n':
            return
            
        while line[i] != b'\0':
            if isdelim(line[i]):
                line[i] = b' '
            i += 1
        
        value = strtof(line, &ptr)
        while True:
            if value > 0:
                peak.mz = value
                peak.intensity = strtof(ptr, &ptr2)
                peaklist.push_back(peak)
            else:
                break
                
            peaks_read += 1
            if peaks_read >= num_peaks:
                return
                    
            value = strtof(ptr2, &ptr)
            if ptr == ptr2:
                break
                
        if fgets(line, MAX_LINE_SIZE, fp) == NULL:
            return
           
           
cdef tuple read_entry(char name[MAX_VALUE_SIZE], FILE * fp, bint ignore_unknown=False):
    """Read a spectrum entry (params and peaklist) from file
    """
    
    cdef:
        dict params = {'name': name.decode('UTF-8', 'ignore'), 'synonyms': []}
        char *ptr = NULL
        char line[MAX_LINE_SIZE]
        vector[peak_t] peaklist
        size_t pos
        char key[MAX_KEY_SIZE]
        char value[MAX_VALUE_SIZE]
        int num_peaks = 0
        bool in_data = False
    
    while fgets(line, MAX_LINE_SIZE, fp) != NULL:
        # Ignore blank lines
        if line[0] == b'\n' or line[0] == b'\r':
            continue
            
        if in_data:
            if num_peaks > 0:
                read_data(line, &peaklist, fp, num_peaks)

            if peaklist.size() > 0:
                return params, arr_from_peaks_vector(peaklist)
            else:
                return params, np.empty((0, 2), dtype=np.float32)
        else:
            ptr = strchr(line, b':')
            if ptr > line:
                if strncasecmp(line, 'NUM PEAKS:', 10) == 0:
                    num_peaks = strtol(line+10, &ptr, 10)
                    in_data = True
                elif strncasecmp(line, 'MW:', 3) == 0:
                    params['mw'] = strtof(line+3, NULL)
                elif strncasecmp(line, 'PRECURSORMZ:', 12) == 0:
                    params['precursormz'] = strtof(line+13, NULL)
                elif strncasecmp(line, 'EXACTMASS:', 10) == 0:
                    params['exactmass'] = strtof(line+11, NULL)
                elif strncasecmp(line, 'RETENTIONTIME:', 14) == 0:
                    params['retentiontime'] = strtof(line+14, NULL)
                elif strncasecmp(line, 'SYNONYM:', 8) == 0:
                    strcpy(value, line+8)
                    pos = strcspn(value, '\r\n')
                    value[pos] = b'\0'
                    trim(value)
                    params['synonyms'].append(value.decode('UTF-8', 'ignore'))
                elif not ignore_unknown:
                    pos = ptr - line
                    strncpy(key, line, pos)
                    key[pos] = b'\0'
                    strlwr(key)
                    strcpy(value, line+pos+1)
                    if strlen(value) > 0:
                        pos = strcspn(value, '\r\n')
                        value[pos] = b'\0'
                        trim(value)
                        params[key.decode('UTF-8', 'ignore')] = value.decode('UTF-8', 'ignore')
            
    if params:
        return params, np.empty((0, 2), dtype=np.float32)
                

def read(str filename, bint ignore_unknown=False):
    cdef:
        bytes fname_bytes
        char *fname
        tuple entry
        char line[MAX_LINE_SIZE]
        FILE *fp
        char name[MAX_VALUE_SIZE]

    fname_bytes = filename.encode(CHARSET)
        
    fname = fname_bytes
        
    fp = fopen(fname, 'r')
    if fp == NULL:
        raise FileNotFoundError(errno.ENOENT,
                                os.strerror(errno.ENOENT),
                                filename)

    while fgets(line, MAX_LINE_SIZE, fp) != NULL:
        if strncasecmp(line, 'NAME:', 5) == 0:
            strcpy(name, line+6)
            name[strcspn(name, '\r\n')] = b'\0'
            entry = read_entry(name, fp, ignore_unknown)
            if entry:
                yield entry
            
    fclose(fp)
