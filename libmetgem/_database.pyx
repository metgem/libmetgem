# cython: language_level=3
# distutils: language=c++

cimport cython
from libc.string cimport const_char, strcpy
from libc.stdio cimport const_void, sprintf
from libc.stdint cimport int64_t
from libc.time cimport clock_t, clock, CLOCKS_PER_SEC
from libc.math cimport fabs
from libcpp cimport bool
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np

from ._filter cimport filter_data_nogil
from ._common cimport peak_t, np_arr_pointer
from ._score cimport generic_score_nogil
from ._common cimport (score_algorithm_t,
                       str_to_score_algorithm,
                       norm_method_t,
                       str_to_norm_method)

from sqlite3 import (InternalError, DataError, DatabaseError, OperationalError,
                     IntegrityError, ProgrammingError)

ctypedef struct db_result_t:
    double score
    int id
    int bank_id
    char name[1024]

ctypedef struct query_result_t:
    int res_code
    char err_msg[48]
    unordered_map[int, vector[db_result_t]] results
    
cdef extern from *:
    '''
    #ifdef _WIN32
        #define CHARSET "mbcs"
    #else
        #define CHARSET "UTF-8"
    #endif
    '''
    extern const char* CHARSET

cdef extern from 'sqlite3.h' nogil:
    ctypedef int sqlite3
    ctypedef int sqlite3_stmt
    ctypedef int sqlite3_context
    ctypedef int sqlite3_value
    ctypedef int64_t sqlite3_int64

    const_char *sqlite3_errmsg(sqlite3*)
    int sqlite3_prepare_v2(sqlite3 * db,
                           char * zSql,
                           int nByte,
                           sqlite3_stmt ** ppStmt,
                           char ** pzTail)
    int sqlite3_step(sqlite3_stmt *)
    int sqlite3_column_type(sqlite3_stmt*, int iCol)
    const unsigned char *sqlite3_column_text(sqlite3_stmt*, int iCol)
    double sqlite3_column_double(sqlite3_stmt * , int iCol)
    sqlite3_int64 sqlite3_column_int(sqlite3_stmt*, int iCol)
    const_void *sqlite3_column_blob(sqlite3_stmt * , int iCol)
    int sqlite3_column_bytes(sqlite3_stmt*, int iCol)
    int sqlite3_finalize(sqlite3_stmt * pStmt)
    int sqlite3_close(sqlite3*)
    int sqlite3_open_v2(const_char *filename, sqlite3 **ppDb,
                        int flags, const_char *zVfs)
    int sqlite3_bind_double(sqlite3_stmt*, int, double)
    int sqlite3_bind_int(sqlite3_stmt*, int, int)
    int sqlite3_bind_text(sqlite3_stmt*,int,const char*,int,void(*)(void*))
    int sqlite3_extended_result_codes(sqlite3*, int onoff)
    int sqlite3_reset(sqlite3_stmt *pStmt)
    
    enum:
        SQLITE_OK
        SQLITE_ERROR
        SQLITE_INTERNAL
        SQLITE_PERM
        SQLITE_ABORT
        SQLITE_BUSY
        SQLITE_LOCKED
        SQLITE_NOMEM
        SQLITE_READONLY
        SQLITE_INTERRUPT
        SQLITE_IOERR       
        SQLITE_CORRUPT
        SQLITE_NOTFOUND
        SQLITE_FULL
        SQLITE_CANTOPEN
        SQLITE_PROTOCOL
        SQLITE_EMPTY
        SQLITE_SCHEMA
        SQLITE_TOOBIG
        SQLITE_CONSTRAINT
        SQLITE_MISMATCH
        SQLITE_MISUSE
        SQLITE_NOLFS
        SQLITE_AUTH
        SQLITE_FORMAT
        SQLITE_RANGE
        SQLITE_NOTADB
        SQLITE_NOTICE
        SQLITE_WARNING
        SQLITE_ROW
        SQLITE_DONE

    enum:
        SQLITE_INTEGER
        SQLITE_FLOAT
        SQLITE_TEXT
        SQLITE_BLOB
        SQLITE_NULL

    enum:
        SQLITE_OPEN_READONLY
        SQLITE_OPEN_READWRITE
        SQLITE_OPEN_CREATE
        SQLITE_OPEN_DELETEONCLOSE
        SQLITE_OPEN_EXCLUSIVE
        SQLITE_OPEN_AUTOPROXY
        SQLITE_OPEN_URI
        SQLITE_OPEN_MEMORY
        SQLITE_OPEN_MAIN_DB
        SQLITE_OPEN_TEMP_DB
        SQLITE_OPEN_TRANSIENT_DB
        SQLITE_OPEN_MAIN_JOURNAL
        SQLITE_OPEN_TEMP_JOURNAL
        SQLITE_OPEN_SUBJOURNAL
        SQLITE_OPEN_MASTER_JOURNAL
        SQLITE_OPEN_NOMUTEX
        SQLITE_OPEN_FULLMUTEX
        SQLITE_OPEN_SHAREDCACHE
        SQLITE_OPEN_PRIVATECACHE
        SQLITE_OPEN_WAL

    enum:
        SQLITE_UTF8
        SQLITE_UTF16LE
        SQLITE_UTF16BE
        SQLITE_UTF16
        SQLITE_ANY
        SQLITE_UTF16_ALIGNED
        
    enum:
        SQLITE_DETERMINISTIC
        
    enum:
        SQLITE_LIMIT_LENGTH
        SQLITE_LIMIT_SQL_LENGTH
        SQLITE_LIMIT_COLUMN
        SQLITE_LIMIT_EXPR_DEPTH
        SQLITE_LIMIT_COMPOUND_SELECT
        SQLITE_LIMIT_VDBE_OP
        SQLITE_LIMIT_FUNCTION_ARG
        SQLITE_LIMIT_ATTACHED
        SQLITE_LIMIT_LIKE_PATTERN_LENGTH
        SQLITE_LIMIT_VARIABLE_NUMBER
        SQLITE_LIMIT_TRIGGER_DEPTH
        SQLITE_LIMIT_WORKER_THREADS
   
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef query_result_t query_nogil(char *fname, vector[int] indices,
                                vector[double] mzvec, vector[peak_t *] datavec,
                                vector[np.npy_intp] data_sizes, vector[int] databases,
                                double mz_tolerance, int min_matched_peaks,
                                int min_intensity, int parent_filter_tolerance,
                                int matched_peaks_window,
                                int min_matched_peaks_search, double min_score,
                                double analog_mz_tolerance=0.,
                                bool positive_polarity=True,
                                double mz_min = 50.,
                                score_algorithm_t score_algorithm = score_algorithm_t.cosine,
                                bool square_root=True,
                                norm_method_t norm_method=norm_method_t.dot,
                                object callback=None) noexcept nogil:
    cdef:
        sqlite3 *db
        sqlite3_stmt *stmt
        int bank_id
        double pepmass
        const peak_t *blob
        int blob_size
        size_t size = mzvec.size()
        char query[4096]
        double score
        int i
        unsigned int j
        int ret
        double mz_low = mzvec[0], mz_high = mzvec[0]
        bool has_callback = callback is not None
        db_result_t r
        query_result_t qr
        vector[peak_t] filtered
        vector[int] ids
        int rows = 0, max_rows = 0
        clock_t t
        char dbs[2048]
        int num_dbs
              
    # Open database
    ret = sqlite3_open_v2(fname, &db, SQLITE_OPEN_READONLY, NULL)

    if db == NULL:
        strcpy(qr.err_msg, sqlite3_errmsg(db))
        qr.res_code = SQLITE_NOMEM
        return qr
    elif ret != SQLITE_OK:
        strcpy(qr.err_msg, sqlite3_errmsg(db))
        sqlite3_close(db)
        qr.res_code = ret
        return qr
    
    sqlite3_extended_result_codes(db, 1)
    
    # Get min/max mz values in list
    for i in range(size):
        if mzvec[i] < mz_low:
            mz_low = mzvec[i]
        elif mzvec[i] > mz_high:
            mz_high = mzvec[i]
               
    # Prepare SQL query
    num_dbs = <int> databases.size()
    if num_dbs > 0:
        # Filter databases
        ret = 0
        for bank_id in databases:
            dbs[ret] = b'?'
            dbs[ret+1] = b','
            ret += 2
        dbs[ret-1] = b'\0'
        sprintf(query, "SELECT id, pepmass, name, peaks, bank_id FROM spectra WHERE bank_id IN (%s) AND (positive = ? OR positive IS NULL) AND PEPMASS BETWEEN ? AND ?", dbs)

        ret = sqlite3_prepare_v2(db, query, -1, &stmt, NULL)
        for i in range(num_dbs):
            sqlite3_bind_int(stmt, i+1, databases[i])
    else:
        ret = sqlite3_prepare_v2(db, "SELECT id, pepmass, name, peaks, bank_id FROM spectra WHERE (positive = ? OR positive IS NULL) AND PEPMASS BETWEEN ? AND ?", -1, &stmt, NULL)

    sqlite3_bind_int(stmt, num_dbs+1, positive_polarity)
    if analog_mz_tolerance > 0:
        sqlite3_bind_double(stmt, num_dbs+2, mz_low-analog_mz_tolerance)
        sqlite3_bind_double(stmt, num_dbs+3, mz_high+analog_mz_tolerance)
    else:
        sqlite3_bind_double(stmt, num_dbs+2, mz_low-mz_tolerance)
        sqlite3_bind_double(stmt, num_dbs+3, mz_high+mz_tolerance)

    if ret != SQLITE_OK:
        strcpy(qr.err_msg, sqlite3_errmsg(db))
        sqlite3_close(db)
        qr.res_code = ret
        return qr
                
    # Get number of results
    t = clock()
    with gil:
        while True:
            ret = sqlite3_step(stmt)
            if ret == SQLITE_DONE:
                break
            elif ret == SQLITE_BUSY:
                continue
            elif ret == SQLITE_ROW:
                max_rows += 1
                
    if max_rows == 0:
        qr.res_code = SQLITE_OK
        return qr
    
    # Reset statement
    ret = sqlite3_reset(stmt)
    if ret != SQLITE_OK:
        strcpy(qr.err_msg, sqlite3_errmsg(db))
        sqlite3_close(db)
        qr.res_code = ret
        return qr
    
    # Loop on results
    t = clock()
    qr.results.reserve(size)
    while True:
        ret = sqlite3_step(stmt)
        if ret == SQLITE_DONE:
            break
        elif ret == SQLITE_BUSY:
            continue
        elif ret == SQLITE_ROW:
            pepmass = sqlite3_column_double(stmt, 1)
            
            ids.clear()
            ids.reserve(size)
            if analog_mz_tolerance > 0:
                for i in range(size):
                    if mzvec[i]-analog_mz_tolerance<=pepmass<=mzvec[i]+analog_mz_tolerance and not (mzvec[i]-mz_tolerance<=pepmass<=mzvec[i]+mz_tolerance) :
                        ids.push_back(i)
            else:
                for i in range(size):
                    if mzvec[i]-mz_tolerance<=pepmass<=mzvec[i]+mz_tolerance:
                        ids.push_back(i)
                    
            if ids.size() > 0:
                blob = <const peak_t*>sqlite3_column_blob(stmt, 3)
                blob_size = sqlite3_column_bytes(stmt, 3) / sizeof(peak_t)
                
                if blob_size > 0:
                    filtered = filter_data_nogil(pepmass, blob, blob_size,
                                             min_intensity, parent_filter_tolerance,
                                             matched_peaks_window, min_matched_peaks_search,
                                             mz_min, square_root, norm_method)
                    for i in ids:
                        score = generic_score_nogil(
                            pepmass, filtered.data(), filtered.size(),
                            mzvec[i], datavec[i], data_sizes[i],
                            mz_tolerance, min_matched_peaks, score_algorithm)
                        if score > min_score:
                            r.score = score
                            r.id = sqlite3_column_int(stmt, 0)
                            r.bank_id = sqlite3_column_int(stmt, 4)
                            if sqlite3_column_type(stmt, 2) == SQLITE_TEXT:
                                strcpy(r.name, <char *>sqlite3_column_text(stmt, 2))
                            else:
                                strcpy(r.name, "Unknown")
                            qr.results[indices[i]].push_back(r)
                            
            rows += 1
            if has_callback and (clock()-t)/<float>CLOCKS_PER_SEC > 0.02:
                t = clock()
                with gil:
                    if not callback(<float>rows/max_rows*100):
                        qr.res_code = -1
                        return qr
        else:
            strcpy(qr.err_msg, sqlite3_errmsg(db))
            sqlite3_close(db)
            qr.res_code = ret
            return qr

    sqlite3_finalize(stmt)
    sqlite3_close(db)
    
    # Free memory
    filtered.clear()
    filtered.shrink_to_fit()
    ids.clear()
    ids.shrink_to_fit()
    
    if has_callback:
        with gil:
            callback(100)
    
    qr.res_code = SQLITE_OK
    return qr
   
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def query(str filename, vector[int] indices, vector[double] mzvec, list datavec,
          vector[int] databases, double mz_tolerance, int min_matched_peaks,
          int min_intensity, int parent_filter_tolerance,
          int matched_peaks_window, int min_matched_peaks_search,
          double min_score, double analog_mz_tolerance=0.,
          bool positive_polarity=True, double mz_min = 50.,
          score_algorithm: str = 'cosine',
          square_root: bool = True, norm: str = 'dot',
          object callback=None):
    cdef:
        bytes fname_bytes
        char *fname
        int res_code
        np.ndarray[np.float32_t, ndim=2] tmp_array
        size_t size = mzvec.size()
        int i
        vector[peak_t *] data_p
        vector[np.npy_intp] data_sizes
        query_result_t qr
        vector[db_result_t] results
        db_result_t r
        score_algorithm_t algorithm = str_to_score_algorithm(score_algorithm)
        norm_method_t norm_method = str_to_norm_method(norm)
        
    data_p.resize(size)
    data_sizes.resize(size)
    for i, tmp_array in enumerate(datavec):
        data_p[i] = <peak_t*>np_arr_pointer(tmp_array)
        data_sizes[i] = tmp_array.shape[0]
        
    fname_bytes = filename.encode(CHARSET)
    fname = fname_bytes
    
    qr = query_nogil(fname, indices, mzvec, data_p, data_sizes, databases,
                     mz_tolerance, min_matched_peaks, min_intensity,
                     parent_filter_tolerance, matched_peaks_window,
                     min_matched_peaks_search, min_score,
                     analog_mz_tolerance, positive_polarity,
                     mz_min, algorithm, square_root, norm_method, callback)
                     
    # Free memory
    data_sizes.clear()
    data_sizes.shrink_to_fit()
                           
    if qr.res_code == SQLITE_OK:
        return qr.results
    elif qr.res_code == -1: # User canceled the process
        return
    elif qr.res_code in (SQLITE_INTERNAL, SQLITE_NOTFOUND):
        raise InternalError(qr.err_msg)
    elif qr.res_code == SQLITE_NOMEM:
        raise MemoryError()
    elif qr.res_code in (SQLITE_ERROR, SQLITE_PERM, SQLITE_ABORT, SQLITE_BUSY,
                         SQLITE_LOCKED, SQLITE_READONLY, SQLITE_INTERRUPT,
                         SQLITE_IOERR, SQLITE_FULL, SQLITE_CANTOPEN,
                         SQLITE_PROTOCOL, SQLITE_EMPTY, SQLITE_SCHEMA):
        raise OperationalError(qr.err_msg)
    elif qr.res_code == SQLITE_CORRUPT:
        raise DatabaseError(qr.err_msg)
    elif qr.res_code == SQLITE_TOOBIG:
        raise DataError(qr.err_msg)
    elif qr.res_code in (SQLITE_CONSTRAINT, SQLITE_MISMATCH):
        raise IntegrityError(qr.err_msg)
    elif qr.res_code == SQLITE_MISUSE:
        raise ProgrammingError(qr.err_msg)
    else:
        raise DatabaseError(qr.err_msg)
