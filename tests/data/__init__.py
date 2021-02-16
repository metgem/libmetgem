import os
import itertools

import numpy as np
import pytest
import warnings

from libmetgem.common import MZ, INTENSITY
from funcs import compute_similarity_matrix_f, kneighbors_graph_from_similarity_matrix_f

__all__ = ('known_filtered_spectrum', 'known_spectrum'
           'known_spectrum_filter_comparison',
           'known_spectra_filter_comparison',
           'known_cosines', 'known_spectra_comparisons',
           'random_spectrum', 'another_random_spectrum', 'random_spectra',
           'mz_tolerance', 'min_matched_peaks',
           'min_intensity', 'parent_filter_tolerance',
           'matched_peaks_window', 'min_matched_peaks_search',
           'pairs_min_cosine', 'top_k',
           'valid_mgf', 'invalid_mgf', 'empty_mgf', 'noions_mgf',
           'valid_msp', 'invalid_msp', 'empty_msp', 'noions_msp',
           'matrix', 'sparse_matrix', 'neighbors_graph', 'random_matrix')

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

SPECTRA_COMPARISONS = {
    (0, 1): (0.02, [(0, 0, 0.6465, 0), (5, 1, 0.0473, 0), (7, 3, 0.0209, 0), (14, 4, 0.0102, 0)]),
    (2, 3): (0.02, [(0, 0, 0.1763, 0), (1, 1, 0.1346, 0), (2, 2, 0.1343, 0), (3,  3, 0.1100, 0), (4, 4, 0.0989, 0),
                    (5, 5, 0.0961, 0), (6, 6, 0.0951, 0), (7, 7, 0.0704, 0), (8,  8, 0.0426, 0), (9, 9, 0.0417, 0)]),
    (4, 5): (0.02, [(0, 1, 0.3187, 0), (2, 4, 0.0576, 1), (1, 6, 0.0572, 0), (18, 0, 0.0438, 1), (6, 3, 0.0358, 1),
                    (17, 5, 0.0211, 1)]),
    (0, 2): (0.02, [])
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
    
@pytest.fixture(params=SPECTRA_COMPARISONS.keys(), scope="session")           
def known_spectra_comparisons(request):
    if request.param in SPECTRA_COMPARISONS:
        return (MZS[request.param[0]],
                MZS[request.param[1]],
                SPECTRA[request.param[0]],
                SPECTRA[request.param[1]],
                *SPECTRA_COMPARISONS[request.param])
           
           
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
        content.append("FORMULA=C11H22NO4")
        content.append("PEPMASS={}".format(MZS[i]))
        for mz, intensity in SPECTRA_UNFILTERED[i]:
            content.append("{} {}".format(mz, intensity))
        content.append("END IONS")
        content.append("")
        
    # Add empty lines at thend, they should be ignored
    content.append("")
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
    content = ["BEGIN IONS",
               "PEPMASS=415.2986",
               "END IONS"]
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
    mz_zeroed = False
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
                elif request.param == "mz-zero" and not mz_zeroed:
                    line = "0\t" + line.split()[1]
                    mz_zeroed = True
        content.append(line)
    p.write("\n".join(content))
    
    return MZS, SPECTRA_UNFILTERED, p
    
    
@pytest.fixture(scope="session")
def valid_msp(tmpdir_factory):
    """Create a valid msp file"""
    
    p = tmpdir_factory.mktemp("valid").join("valid.msp")
    content = []
    for i in range(len(MZS)):
        content.append("NAME: {}".format(i+1))
        content.append("FORMULA: C11H22NO4")
        content.append("PRECURSORMZ: {}".format(MZS[i]))
        content.append("MW: {}".format(MZS[i]))
        content.append("EXACTMASS: {}".format(MZS[i]))
        content.append("SYNONYM: foo, bar")
        content.append("RETENTIONTIME: 2.58")
        content.append("Num Peaks: {}".format(len(SPECTRA_UNFILTERED[i])))
        for mz, intensity in SPECTRA_UNFILTERED[i]:
            content.append("{} {}".format(mz, intensity))
        content.append("")
        
    # Add empty lines at thend, they should be ignored
    content.append("")
    content.append("")
    
    p.write("\n".join(content))    
    
    return MZS, SPECTRA_UNFILTERED, p
    

@pytest.fixture(scope="session")
def empty_msp(tmpdir_factory):
    """Creates an empty msp file"""
        
    p = tmpdir_factory.mktemp("empty").join("empty.msp")
    p.write("")
    return p
    
    
@pytest.fixture(scope="session")
def noions_msp(tmpdir_factory):
    """Creates a msp file with an entry where ions are missing"""
    
    p = tmpdir_factory.mktemp("noions").join("noions.msp")
    content = ["NAME: test",
               "PRECURSORMZ: 415.2986",
               "Num Peaks: 2"]
    p.write("\n".join(content))
    
    return p
    
    
@pytest.fixture()
def invalid_msp(tmpdir_factory, valid_msp, request):
    """Create an invalid msp file from `valid_msp`. Invalidity can
        be parametrized using indirection."""
        
    _, _, mgf = valid_msp
        
    if not hasattr(request, 'param') or request.param is None:
        return valid_mgf
    
    p = tmpdir_factory.mktemp("invalid", numbered=True).join("invalid.msp")
    content = []
    i = -1
    mz_zeroed = False
    for line in mgf.read().split("\n"):
        if line.startswith("NAME:"):
            i += 1
        elif i==2:
            if line.startswith("PRECURSORMZ:"):
                if request.param == "semicolon":
                    line = line.replace(":", "=")
                elif request.param == "no-pepmass":
                    line = ""
                elif request.param == "wrong-float":
                    line += "f"
            elif line.startswith("Num Peaks:") and request.param == "no-num-peaks":
                line = ""
            elif request.param == "two-names":
                line = "NAME: Second name"
            elif line.startswith("Num Peaks:") and request.param == "num-peaks-zero":
                line = "Num Peaks: 0"
            elif line and line[0].isnumeric() and request.param == "mz-zero" and not mz_zeroed:
                line = "0\t" + line.split()[1]
                mz_zeroed = True

        content.append(line)
    p.write("\n".join(content))
    
    return MZS, SPECTRA_UNFILTERED, p
    
    
@pytest.fixture(params=list(itertools.product(MZ_TOLERANCES, MIN_MATCHED_PEAKS)),
                scope='session')
def matrix(request, random_spectra, compute_similarity_matrix_f):
    """Compute similarity matrix for different `mz_tolerance` and
        `min_matched_peaks` combinations.
    """
    
    mzs, spectra = random_spectra
    m = compute_similarity_matrix_f(mzs, spectra, request.param[0], request.param[1], dense_output=True)
    m.flags.writeable = False
    
    return m
    
    
@pytest.fixture(params=list(itertools.product(MZ_TOLERANCES, MIN_MATCHED_PEAKS)),
                scope='session')
def sparse_matrix(request, random_spectra, compute_similarity_matrix_f):
    """Compute similarity matrix for different `mz_tolerance` and
        `min_matched_peaks` combinations.
    """
    
    mzs, spectra = random_spectra
    m = compute_similarity_matrix_f(mzs, spectra, request.param[0], request.param[1], dense_output=False)
    
    return m
    
    
@pytest.fixture(params=[0, 5, 9],
                scope='session')
def neighbors_graph(request, sparse_matrix, kneighbors_graph_from_similarity_matrix_f):
    """Compute neighbors graph for different matrix and n_neighbors combinations."""
    m = kneighbors_graph_from_similarity_matrix_f(sparse_matrix, n_neighbors=request.param)
    
    return request.param, m
    
    
@pytest.fixture(params=range(10), scope="session")
def random_matrix(request):
    """Creates a few random similarity matrix.
    """
    
    np.random.seed(request.param)
    scores = np.triu(np.random.random((50,50)).astype(dtype=np.float32))
    scores = scores + scores.T
    np.fill_diagonal(scores, 1)
    
    # Make data array immutable
    scores.flags.writeable = False
    
    mzs = list(np.random.random((50,)) * 1000 + 50)
    
    return mzs, scores
