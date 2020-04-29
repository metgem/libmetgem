"""
Test `libmetgem.msp module
"""

import pytest
import numpy as np


from libmetgem import IS_CYTHONIZED, MZ, INTENSITY
from libmetgem.msp import read as read_msp

from data import valid_msp, invalid_msp, empty_msp, noions_msp
  
    
@pytest.mark.parametrize('ignore_unknown', [True, False])
def test_msp_pepmass(valid_msp, ignore_unknown):
    """`read` should return the same *m/z* values than the ones written in msp.
    """
    
    mzs, _, p = valid_msp
    gen = read_msp(str(p), ignore_unknown=ignore_unknown)
    for i, (params, _) in enumerate(gen):
        assert pytest.approx(params['precursormz']) == mzs[i]
        
                
@pytest.mark.parametrize('ignore_unknown', [True, False])
def test_msp_data(valid_msp, ignore_unknown):
    """`read` should return the same data arrays than the ones written in msp.
    """
    
    _, spectra, p = valid_msp
    gen = read_msp(str(p), ignore_unknown=ignore_unknown)
    for i, (_, data) in enumerate(gen):
        assert pytest.approx(data) == spectra[i]


@pytest.mark.python
@pytest.mark.skipif(getattr(read_msp, '__wrapped__', None) is None,
                    reason="libmetgem should be cythonized")            
@pytest.mark.parametrize('ignore_unknown', [True, False])
def test_msp_ignore_unknown(valid_msp, ignore_unknown):
    """`ignore_unknown` parameter should never ignore precursormz.
    """
    
    _, _, p = valid_msp
    gen = read_msp(str(p), ignore_unknown=ignore_unknown)
    for i, (params, _) in enumerate(gen):
        assert 'precursormz' in params
        if ignore_unknown:
            assert 'formula' not in params
        else:
            assert 'formula' in params
        

@pytest.mark.parametrize('ignore_unknown', [True, False])
def test_msp_empty(empty_msp, ignore_unknown):
    """An empty msp file should not throw an error but yield an empty list
    """
    
    data = list(read_msp(str(empty_msp), ignore_unknown=ignore_unknown))
    assert isinstance(data, list)
    assert len(data) == 0
    
    
@pytest.mark.parametrize('filename', ["", "ddff", "fgkgkrk"])
@pytest.mark.parametrize('ignore_unknown', [True, False])
def test_msp_non_existing(filename, ignore_unknown):
    """A filename that does not exists should throw a FileNotFoundError
    """
    
    with pytest.raises(FileNotFoundError):
        data = list(read_msp(filename, ignore_unknown=ignore_unknown))
    
    
@pytest.mark.parametrize('ignore_unknown', [True, False])
@pytest.mark.parametrize('invalid_msp', ["semicolon", "no-pepmass"],
    indirect=True)
def test_msp_invalid(invalid_msp, ignore_unknown):
    """Invalid precursormz declaration should be ignored.
    """
    
    _, _, p = invalid_msp
    gen = read_msp(str(p), ignore_unknown=ignore_unknown)
    for i, (params, data) in enumerate(gen):
        if i == 2:
            assert 'precursormz' not in params
            
 
@pytest.mark.parametrize('ignore_unknown', [True, False])
@pytest.mark.parametrize('invalid_msp', ["no-num-peaks"],
    indirect=True)
def test_msp_nonumpeaks(invalid_msp, ignore_unknown):
    """Entries where num peaks field is missing should be ignored.
    """
    
    mzs, _, p = invalid_msp
    data = list(read_msp(str(p), ignore_unknown=ignore_unknown))
    assert len(data) == len(mzs) - 1
    for i, (params, d) in enumerate(data):
        if i < 2:
            assert pytest.approx(params['precursormz']) == mzs[i]
        else:
            assert pytest.approx(params['precursormz']) == mzs[i+1]
       
       
@pytest.mark.parametrize('ignore_unknown', [True, False])
def test_msp_noions(noions_msp, ignore_unknown):
    """A msp file with only one entry where there is no ions should not throw an
       error.
    """

    data = list(read_msp(str(noions_msp), ignore_unknown=ignore_unknown))
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0][1].size == 0
    assert data[0][1].shape == (0, 2)
            

@pytest.mark.python
@pytest.mark.skipif(not IS_CYTHONIZED, reason="libmetgem should be cythonized")
@pytest.mark.parametrize('ignore_unknown', [True, False])
def test_msp_python_cython(valid_msp, ignore_unknown):
    """Cythonized `read_msp` and it's fallback Python version should give the
       same results.
    """
    
    _, _, p = valid_msp
    
    gen_p = read_msp.__wrapped__(str(p), ignore_unknown=ignore_unknown)
    gen_c = read_msp(str(p), ignore_unknown=ignore_unknown)
    for (params_p, data_p), (params_c, data_c) in zip(gen_p, gen_c):
        assert params_p.keys () == params_c.keys()
        for key in params_p.keys():
            if isinstance(params_p[key], float):
                assert pytest.approx(params_p[key]) == params_c[key]
            else:
                assert params_p[key] == params_c[key]
        assert pytest.approx(data_p) == data_c
