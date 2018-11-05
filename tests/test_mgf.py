"""
Test `libmetgem.mgf` module
"""

import pytest
import numpy as np


from libmetgem import IS_CYTHONIZED, MZ, INTENSITY
from libmetgem.mgf import read as read_mgf

from data import valid_mgf, invalid_mgf, empty_mgf, noions_mgf
  
    
@pytest.mark.parametrize('ignore_unknown', [True, False])
def test_mgf_pepmass(valid_mgf, ignore_unknown):
    """`read` should return the same *m/z* values than the ones written in mgf.
    """
    
    mzs, _, p = valid_mgf
    gen = read_mgf(str(p), ignore_unknown=ignore_unknown)
    for i, (params, _) in enumerate(gen):
        assert params['pepmass'] == mzs[i]
        
        
@pytest.mark.parametrize('ignore_unknown', [True, False])
@pytest.mark.parametrize('invalid_mgf', ["comma-pepmass"], indirect=True)
def test_mgf_pepmass_comma(invalid_mgf, ignore_unknown):
    """Comma can be used as decimal separator for `pepmass`"""
    
    mzs, _, p = invalid_mgf
    gen = read_mgf(str(p), ignore_unknown=ignore_unknown)
    for i, (params, _) in enumerate(gen):
        assert params['pepmass'] == mzs[i]

        
@pytest.mark.parametrize('ignore_unknown', [True, False])
def test_mgf_data(valid_mgf, ignore_unknown):
    """`read` should return the same data arrays than the ones written in mgf.
    """
    
    _, spectra, p = valid_mgf
    gen = read_mgf(str(p), ignore_unknown=ignore_unknown)
    for i, (_, data) in enumerate(gen):
        assert pytest.approx(data) == spectra[i]

        
@pytest.mark.parametrize('ignore_unknown', [True, False])
@pytest.mark.parametrize('invalid_mgf', ["comma-data"], indirect=True)
def test_mgf_data_comma(invalid_mgf, ignore_unknown):
    """Comma use as a separator in data arrays will lead to truncated floats.
    """
    
    _, spectra, p = invalid_mgf
    gen = read_mgf(str(p), ignore_unknown=ignore_unknown)
    for i, (_, data) in enumerate(gen):
        if i == 2:
            assert data[:, MZ] == pytest.approx(spectra[i][:, MZ], abs=1)
            assert np.count_nonzero(data[:, INTENSITY]) == 0


@pytest.mark.python
@pytest.mark.skipif(getattr(read_mgf, '__wrapped__', None) is None,
                    reason="libmetgem should be cythonized")            
@pytest.mark.parametrize('ignore_unknown', [True, False])
def test_mgf_ignore_unknown(valid_mgf, ignore_unknown):
    """`ignore_unknown` parameter should never ignore pepmass.
    """
    
    _, _, p = valid_mgf
    gen = read_mgf(str(p), ignore_unknown=ignore_unknown)
    for i, (params, _) in enumerate(gen):
        assert 'pepmass' in params
        if ignore_unknown:
            assert 'feature_id' not in params
        else:
            assert 'feature_id' in params
        

@pytest.mark.parametrize('ignore_unknown', [True, False])
def test_mgf_empty(empty_mgf, ignore_unknown):
    """An empty mgf file should not throw an error but yield an empty list
    """
    
    data = list(read_mgf(str(empty_mgf), ignore_unknown=ignore_unknown))
    assert isinstance(data, list)
    assert len(data) == 0
    
    
@pytest.mark.parametrize('filename', ["", "ddff", "fgkgkrk"])
@pytest.mark.parametrize('ignore_unknown', [True, False])
def test_mgf_non_existing(filename, ignore_unknown):
    """A filename that does not exists should throw a FileNotFoundError
    """
    
    with pytest.raises(FileNotFoundError):
        data = list(read_mgf(filename, ignore_unknown=ignore_unknown))
    
    
@pytest.mark.parametrize('ignore_unknown', [True, False])
@pytest.mark.parametrize('invalid_mgf', ["semicolon", "no-pepmass"],
    indirect=True)
def test_mgf_invalid(invalid_mgf, ignore_unknown):
    """Invalid pepmass declaration should be ignored.
    """
    
    _, _, p = invalid_mgf
    gen = read_mgf(str(p), ignore_unknown=ignore_unknown)
    for i, (params, data) in enumerate(gen):
        if i == 2:
            assert 'pepmass' not in params
            
            
@pytest.mark.parametrize('ignore_unknown', [True, False])
def test_mgf_noions(noions_mgf, ignore_unknown):
    """A mgf file with only one entry where there is no ions should not throw an
       error.
    """
       
    data = list(read_mgf(str(noions_mgf), ignore_unknown=ignore_unknown))
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0][1].size == 0
    assert data[0][1].shape == (0, 2)
            

@pytest.mark.python
@pytest.mark.skipif(not IS_CYTHONIZED, reason="libmetgem should be cythonized")
def test_mgf_python_cython(valid_mgf):
    """Cythonized `read_mgf` and it's fallback Python version should give the
       same results.
    """
    
    _, _, p = valid_mgf
    
    # Python version does not use `ignore_unknown` parameter
    gen_p = read_mgf.__wrapped__(str(p), ignore_unknown=False)
    gen_c = read_mgf(str(p), ignore_unknown=False)
    for (params_p, data_p), (params_c, data_c) in zip(gen_p, gen_c):
        assert params_p == params_c
        assert pytest.approx(data_p) == data_c