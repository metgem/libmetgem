"""
    Query a MS/MS spectra database for spectra
"""

from ._loader import load_cython
from .filter import filter_data
from .cosine import cosine_score

import time
import numpy as np
import sqlite3

from typing import List, Callable, Dict, Union


@load_cython
def query(fname: str, indices: List[int], mzvec: List[float],
          datavec: List[np.ndarray], databases: List[int], mz_tolerance: float,
          min_matched_peaks: int, min_intensity: int,
          parent_filter_tolerance: int, matched_peaks_window: int,
          min_matched_peaks_search: int, min_cosine: float,
          analog_mz_tolerance: float=0., positive_polarity: bool=True,
          callback: Callable[[int], bool]=None) -> Dict[int, List[Dict[str, Union[float, int]]]]:
    """
        Query an SQLite database containing MS/MS spectra for either standards
        (spectra with similar parent ion's *m/z* and similar MS/MS spectrum) or 
        analogs (spectra with different parent ion's *m/z* but similar MS/MS
        spectrum).
    
    Args:
        fname: filename of database to query.
        indices: list of int identifying data in `mzvec` and `datavec`
        mzvec: list of *m/z* of MS/MS parent ions.
        datavec: list of 2D array containing MS/MS spectra data.
        databases: list of int identifying banks to query.
        mz_tolerance: Maximum *m/z* difference between a spectrum's parent ion
            and a database hit's parent ion to classify the latter as standard.
            Also used for cosine score calculation.
        min_matched_peaks: Used for cosine score calculation.
        min_intensity: Used to filter database spectra.
        parent_filter_tolerance: Used to filter database spectra.
        matched_peaks_window: Used to filter database spectra.
        min_matched_peaks_search: Used to filter database spectra.
        min_cosine: Keeps only hits with a score higher than this value.
        analog_mz_tolerance: Maximum *m/z* difference between a spectrum's
            parent ion and a database hit's parent ion to classify the latter
            as analog. If *m/z* delta is lower than `mz_tolerance`, database
            result is considered to be a standard.
        positive_polarity: If True, only spectra with a positive or undefined
            polarity will be considered, otherwise look only for negative or
            undefined polarity.
        callback: function called to track progress of query. First parameter
            (`int`) is the number of rows evaluated since last call. It should
            return True if processing should continue, or False if query
            should be aborted.
    
    Returns:
        A dictionary with integers as keys: the `indices` for which a result
        has been found. Values are lists of dictionaries containing score
        between spectrum and the databases standard/analog, hit's id in the
        database, id of bank where hit was found and hit's name.
        
    See Also:
        cosine_score, filter_data
    
    """

    size = len(mzvec)

    conn = sqlite3.connect(f'file:{fname}?mode=ro', uri=True)

    # Get min/max mz values in list
    mz_min = min(mzvec)
    mz_max = max(mzvec)

    # Set tolerance
    tol = analog_mz_tolerance if analog_mz_tolerance > 0 else mz_tolerance

    if len(databases) > 0:
        dbs = ','.join([str(x) for x in databases])
        c = conn.execute("SELECT id, pepmass, name, peaks, bank_id FROM spectra WHERE bank_id IN (?4) AND (positive = ?1 OR positive IS NULL) AND PEPMASS BETWEEN ?2 AND ?3",
                         (positive_polarity, mz_min-tol, mz_max+tol, dbs))
    else:
        c = conn.execute("SELECT id, pepmass, name, peaks, bank_id FROM spectra WHERE (positive = ?1 OR positive IS NULL) AND PEPMASS BETWEEN ?2 AND ?3",
                         (positive_polarity, mz_min-tol, mz_max+tol))

    results = c.fetchall()
    max_rows = len(results)

    # Loop on results
    rows = 0
    qr = {}
    t = time.time()
    for row in results:
        pepmass = row[1]
        ids = []
        if analog_mz_tolerance > 0:
            for i in range(size):
                if mzvec[i] - analog_mz_tolerance <= pepmass <= mzvec[i] + analog_mz_tolerance \
                        and not (mzvec[i] - mz_tolerance <= pepmass <= mzvec[i] + mz_tolerance):
                    ids.append(i)
        else:
            for i in range(size):
                if mzvec[i] - mz_tolerance <= pepmass <= mzvec[i] + mz_tolerance:
                    ids.append(i)

        if len(ids) > 0:
            peaks = np.frombuffer(row[3], dtype='<f4').reshape(-1, 2)
            if len(peaks) > 0:
                filtered = filter_data(pepmass, peaks, min_intensity, parent_filter_tolerance,
                                       matched_peaks_window, min_matched_peaks_search)
                for i in ids:
                    score = cosine_score(pepmass, filtered, mzvec[i], datavec[i],
                                         mz_tolerance, min_matched_peaks)
                    if score > min_cosine:
                        r = {'score': score, 'id': row[0], 'bank_id': row[4], 'name': row[2]}
                        try:
                            qr[indices[i]].append(r)
                        except KeyError:
                            qr[indices[i]] = [r]

        rows += 1
        if callback is not None and time.time() - t > 0.02:
            t = time.time()
            if not callback(rows / max_rows * 100):
                return

    if callback is not None and rows % size != 0:
        callback(100)

    return qr
