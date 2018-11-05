import os
import itertools

import numpy as np
import pytest

from libmetgem.common import MZ, INTENSITY
from libmetgem.cosine import compute_distance_matrix

__all__ = ('known_filtered_spectrum', 'known_spectrum'
           'known_spectrum_filter_comparison',
           'known_spectra_filter_comparison',
           'random_spectrum', 'another_random_spectrum', 'random_spectra',
           'mz_tolerance', 'min_matched_peaks',
           'min_intensity', 'parent_filter_tolerance',
           'matched_peaks_window', 'min_matched_peaks_search',
           'pairs_min_cosine', 'top_k',
           'valid_mgf', 'invalid_mgf', 'empty_mgf', 'noions_mgf',
           'matrix', 'random_matrix')

MZ_TOLERANCES = (0.02, 0.05, 1.0)
MIN_MATCHED_PEAKS = (0, 4, 6)
MIN_INTENSITIES = (0, 10, 100)
PARENT_FILTER_TOLERANCES = (0, 17, 50)
MATCHED_PEAKS_WINDOWS = (0, 50, 100)
MIN_MATCHED_PEAKS_SEARCHS = (0, 6, 12)
PAIRS_MIN_COSINES = (0, 0.3, 0.7)
TOP_KS = (0, 1, 10)


MZS = (723.3371, 885.3909, 643.2885, 643.289, 487.2665, 545.2717)
SPECTRA = {}
SPECTRA_UNFILTERED = {}
dir = os.path.dirname(__file__)
for i, mz in enumerate(MZS):
    SPECTRA[i] = np.load(os.path.join(dir, str(i) + ".npy")) \
        .astype(np.float32)
    SPECTRA[i].flags.writeable = False
    SPECTRA_UNFILTERED[i] = np.load(os.path.join(dir, str(i) + "u.npy")) \
        .astype(np.float32)
    SPECTRA_UNFILTERED[i].flags.writeable = False
        
COSINES = {
    (0, 1): (0.02, 4, 0.7249),
    (2, 3): (0.02, 4, 1.0),
    (4, 5): (0.02, 4, 0.5342),
    (0, 2): (0.02, 4, 0.0)
    }
        
        
@pytest.fixture(params=range(len(MZS)), scope="session")
def known_filtered_spectrum(request):
    return MZS[request.param], SPECTRA[request.param]

    
@pytest.fixture(params=range(len(MZS)), scope="session")
def known_spectrum(request):
    return MZS[request.param], SPECTRA_UNFILTERED[request.param]
    
      
@pytest.fixture(params=range(len(MZS)), scope="session")
def known_spectrum_filter_comparison(request):
    return (MZS[request.param],
           SPECTRA_UNFILTERED[request.param],
           SPECTRA[request.param])
           

@pytest.fixture(params=range(len(MZS)), scope="session")
def known_spectra_filter_comparison(request):
    mzs = []
    spectra = []
    unfiltered_spectra = []
    for i in range(len(MZS)):
        mzs.append(MZS[i])
        unfiltered_spectra.append(SPECTRA_UNFILTERED[i])
        spectra.append(SPECTRA[i])
        
    return mzs, unfiltered_spectra, spectra

           
@pytest.fixture(params=COSINES.keys(), scope="session")           
def known_cosines(request):
    if request.param in COSINES:
        return (MZS[request.param[0]],
                MZS[request.param[1]],
                SPECTRA[request.param[0]],
                SPECTRA[request.param[1]],
                *COSINES[request.param])
           
           
def _random_spectrum(seed=0):
    np.random.seed(seed)
    mzs = np.random.random((6,)) * 1000 + 50
    np.random.seed(seed+100)
    intensitys = np.random.random((6,)) * 1000
    
    data = np.vstack((mzs, intensitys)).T
    parent = np.sort(data[:, MZ])[-2:].mean() + np.random.random() * 50
    
    # Data array should be c-contiguous
    data = np.ascontiguousarray(data).astype(np.float32)
        
    # Intensities should be normalized before computing cosine scores
    data[:, INTENSITY] = data[:, INTENSITY] / np.sqrt(data[:, INTENSITY] @ data[:, INTENSITY])
    
    # Make data array immutable
    data.flags.writeable = False
    
    return parent, data

    
@pytest.fixture(params=range(10), scope="session")
def random_spectrum(request):
    """Creates a fake plausible spectrum with random numbers
    """
    
    return _random_spectrum(request.param)
    
    
@pytest.fixture(params=range(10), scope="session")
def another_random_spectrum(request):
    """Creates another fake spectrum for test which two different
       random spectra.
    """
    
    return _random_spectrum(request.param * 1000)

    
@pytest.fixture(params=range(10), scope="session")
def random_spectra(request):
    """Creates a few random spectra.
    """
    
    mzs = []
    spectra = []
    for i in range(10):
        mz, data = _random_spectrum(request.param+i)
        mzs.append(mz)
        spectra.append(data)
        
    return mzs, spectra

    
@pytest.fixture(params=MZ_TOLERANCES, scope="session")
def mz_tolerance(request):
    """Provides some values for `cosine_score`' `mz_tolerance` parameter.
    """
    
    return request.param
    
    
@pytest.fixture(params=MIN_MATCHED_PEAKS, scope="session")
def min_matched_peaks(request):
    """Provides some values for `cosine_score`' `min_matched_peaks` parameter.
    """
    
    return request.param
    
    
@pytest.fixture(params=MIN_INTENSITIES, scope="session")
def min_intensity(request):
    """Provides some values for `filter_data`' `min_intensity` parameter.
    """
    
    return request.param

    
@pytest.fixture(params=PARENT_FILTER_TOLERANCES, scope="session")
def parent_filter_tolerance(request):
    """Provides some values for `filter_data`' `parent_filter_tolerance` parameter.
    """
    
    return request.param
    
    
@pytest.fixture(params=MATCHED_PEAKS_WINDOWS, scope="session")
def matched_peaks_window(request):
    """Provides some values for `filter_data`' `matched_peak_window` parameter.
    """
    
    return request.param
    
    
@pytest.fixture(params=MIN_MATCHED_PEAKS_SEARCHS, scope="session")
def min_matched_peaks_search(request):
    """Provides some values for `filter_data`' `min_matched_peaks_search`
       parameter.
    """
    
    return request.param
    
    
@pytest.fixture(params=PAIRS_MIN_COSINES, scope="session")
def pairs_min_cosine(request):
    """Provides some values for `generate_network`' `pairs_min_cosine`
       parameter.
    """
    
    return request.param
    
    
@pytest.fixture(params=TOP_KS, scope="session")
def top_k(request):
    """Provides some values for `generate_network`' `top_k`
       parameter.
    """
    
    return request.param
    
    
@pytest.fixture(scope="session")
def valid_mgf(tmpdir_factory):
    """Create a valid mgf file"""
    
    p = tmpdir_factory.mktemp("valid").join("valid.mgf")
    content = []
    for i in range(len(MZS)):
        content.append("BEGIN IONS")
        content.append("FEATURE_ID={}".format(i+1))
        content.append("PEPMASS={}".format(MZS[i]))
        for mz, intensity in SPECTRA_UNFILTERED[i]:
            content.append("{} {}".format(mz, intensity))
        content.append("END IONS")
        content.append("")
    p.write("\n".join(content))
    
    return MZS, SPECTRA_UNFILTERED, p
    

@pytest.fixture(scope="session")
def empty_mgf(tmpdir_factory):
    """Creates an empty mgf file"""
        
    p = tmpdir_factory.mktemp("empty").join("empty.mgf")
    p.write("")
    return p
    
    
@pytest.fixture(scope="session")
def noions_mgf(tmpdir_factory):
    """Creates a mgf file with an entry where ions are missing"""
    
    p = tmpdir_factory.mktemp("noions").join("noions.mgf")
    content = ["BEGIN IONS\n",
               "PEPMASS=415.2986\n",
               "END IONS\n"]
    p.write("\n".join(content))
    
    return p
    
    
@pytest.fixture()
def invalid_mgf(tmpdir_factory, valid_mgf, request):
    """Create an invalid mgf file from `valid_mgf`. Invalidity can
        be parametrized using indirection."""
        
    _, _, mgf = valid_mgf
        
    if not hasattr(request, 'param') or request.param is None:
        return valid_mgf
    
    p = tmpdir_factory.mktemp("invalid", numbered=True).join("invalid.mgf")
    content = []
    i = -1
    for line in mgf.read().split("\n"):
        if line.startswith("BEGIN IONS"):
            i+=1
        elif i==2:
            if line.startswith("PEPMASS="):
                if request.param == "semicolon":
                    line = line.replace("=", ":")
                elif request.param == "no-pepmass":
                    line = ""
                elif request.param == "comma-pepmass":
                    line = line.replace(".", ",")
            elif not line.startswith("END IONS") and not "=" in line:
                if request.param == "comma-data":
                    line = line.replace(".", ",")
        content.append(line)
    p.write("\n".join(content))
    
    return MZS, SPECTRA_UNFILTERED, p
    
    
@pytest.fixture(params=list(itertools.product(MZ_TOLERANCES, MIN_MATCHED_PEAKS)),
                scope='session')
def matrix(request, random_spectra):
    """Compute distance matrix for different `mz_tolerance` and
        `min_matched_peaks` combinations.
    """
    
    mzs, spectra = random_spectra
    m = compute_distance_matrix(mzs, spectra, request.param[0], request.param[1])
    m.flags.writeable = False
    
    return m
    
    
@pytest.fixture(params=range(10), scope="session")
def random_matrix(request):
    """Creates a few random distance matrix.
    """
    
    np.random.seed(request.param)
    scores = np.triu(np.random.random((50,50)).astype(dtype=np.float32))
    scores = scores + scores.T
    np.fill_diagonal(scores, 1)
    
    # Make data array immutable
    scores.flags.writeable = False
    
    mzs = list(np.random.random((50,)) * 1000 + 50)
    
    return mzs, scores