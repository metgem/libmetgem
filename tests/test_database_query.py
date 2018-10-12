"""
Test `libmetgem.database.query`
"""

import pytest
import numpy as np
import sqlite3
import os

from libmetgem import MZ, INTENSITY
from libmetgem.database import query
from libmetgem.filter import filter_data
from libmetgem.cosine import cosine_score

from data import (random_spectra,
                  mz_tolerance, min_matched_peaks, min_intensity,
                  parent_filter_tolerance, matched_peaks_window,
                  min_matched_peaks_search)

                  
@pytest.fixture(scope="session")
def db(tmpdir_factory, random_spectra, request):
    """Creates a database with randomly generated spectra inside.
    """
    
    p = tmpdir_factory.mktemp("db", numbered=True).join("test.sqlite")

    conn = sqlite3.connect(str(p))
    
    c = conn.cursor()
    c.execute("CREATE TABLE banks (id INTEGER PRIMARY KEY, name VARCHAR, UNIQUE (name))")
    c.execute("""CREATE TABLE spectra (id integer PRIMARY KEY,
                 bank_id INTEGER NOT NULL, pepmass FLOAT NOT NULL,
                 name VARCHAR, peaks BLOB, positive BOOLEAN,
                 FOREIGN KEY(bank_id) REFERENCES banks (id))""")
    for i in range(10):
        c.execute("INSERT INTO banks (id, name) VALUES (?, ?)", (i, "bank" + str(i)))
        
    mzs, spectra = random_spectra
    
    q = """INSERT INTO spectra (id, bank_id, pepmass, name, positive, peaks)
           VALUES (?, ?, ?, ?, ?, ?)"""
    for i in range(len(mzs)):
        c.execute(q, (i, 1, mzs[i], "spec" + str(i), True,
                      spectra[i].tobytes(order='C')))
        
    conn.commit()
    conn.close()
    return p, random_spectra
    
   
def test_query_random_spectra(db):
    """Test if looking for a spectra that is for sure in database will
       successfully returns this spectra.
    """
    
    p, (mzs, spectra) = db
    
    for i, (mz, data) in enumerate(zip(mzs, spectra)):
        filtered = filter_data(mz, data, 0, 17, 50, 6)
        results = query(str(p), [i], [mz], [filtered], [],
                        0.02, 0, 0, 17, 50, 6, 0.)
        assert i in results
        seen_i = False
        for r in results[i]:
            assert 'id' in r
            assert 'bank_id' in r
            assert 'name' in r
            assert 'score' in r
            
            if r['id'] == i:
                seen_i = True
                assert pytest.approx(r['score']) == 1.0
        assert seen_i
        

def test_query_analog(db):
    """Build an analog and try to find the original spectrum in the database.
    """
    
    p, (mzs, spectra) = db
    
    for i, (mz, data) in enumerate(zip(mzs, spectra)):
        mz = mzs[i] - 50
        data = spectra[i].copy()
        data[:, MZ] = data[:, MZ] - 50
        
        filtered_analog = filter_data(mz, data, 0, 17, 50, 6)
        filtered_orig = filter_data(mzs[i], spectra[i], 0, 17, 50, 6)
        score = cosine_score(mzs[i], filtered_orig,
                             mz, filtered_analog,
                             0.02, 0)
        results = query(str(p), [i], [mz], [filtered_analog], [],
                        0.02, 0, 0, 17, 50, 6, 0., 100.)
        assert i in results
        seen_i = False
        for r in results[i]:
            assert 'id' in r
            assert 'bank_id' in r
            assert 'name' in r
            assert 'score' in r
                
            if r['id'] == i:
                seen_i = True
                assert r['score'] == pytest.approx(score)
        assert seen_i

        
@pytest.mark.slow
@pytest.mark.python
@pytest.mark.skipif(getattr(query, '__wrapped__', None) is None,
                    reason="libmetgem should be cythonized")
def test_query_python_cython(db):
    """Cythonized `query` and it's fallback Python version should give the same
       results.
    """
    
    p, (mzs, spectra) = db
    
    for i, (mz, data) in enumerate(zip(mzs, spectra)):
        filtered = filter_data(mz, data, 0, 17, 50, 6)
        results_p = query.__wrapped__(str(p), [i], [mz], [filtered], [],
                        0.02, 0, 0, 17, 50, 6, 0.)
        results_c = query(str(p), [i], [mz], [filtered], [],
                        0.02, 0, 0, 17, 50, 6, 0.)
        assert results_p.keys() == results_c.keys()
        for k in results_c.keys():
            for r_c, r_p in zip(results_c[k], results_p[k]):
                assert r_c['id'] == r_p['id']
                assert r_c['bank_id'] == r_p['bank_id']
                assert r_c['name'].decode() == r_p['name']
                assert r_c['score'] == pytest.approx(r_p['score'])
    