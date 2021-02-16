import pytest
import warnings

from libmetgem import IS_CYTHONIZED
from libmetgem.cosine import compute_similarity_matrix
from libmetgem.network import generate_network
from libmetgem.mgf import read as read_mgf
from libmetgem.msp import read as read_msp
from libmetgem.filter import filter_data, filter_data_multi
from libmetgem.cosine import cosine_score, compare_spectra
from libmetgem.database import query
from libmetgem.neighbors import kneighbors_graph_from_similarity_matrix

VARIANTS_TO_TEST = ["python"]
if IS_CYTHONIZED:
    VARIANTS_TO_TEST.append("cython")
    
    
class FuncWrapper:
    def __init__(self, func, variant):
        if IS_CYTHONIZED and variant == 'python':
            try:
                func = func.__wrapped__
            except AttributeError:
                warnings.warn("Cythonized '{}' cannot be loaded.".format(func.__name__))
                
        self._func = func
        self.variant = variant
        
    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)
    
    def __str__(self):
        return "<{} ({})>".format(self._func.__name__, self.variant)
    
    def __repr__(self):
        return self.__str__()
        
  
@pytest.fixture(params=VARIANTS_TO_TEST, scope='session')
def kneighbors_graph_from_similarity_matrix_f(request):
    return FuncWrapper(kneighbors_graph_from_similarity_matrix, request.param)
  
@pytest.fixture(params=VARIANTS_TO_TEST, scope='session')
def compute_similarity_matrix_f(request):
    return FuncWrapper(compute_similarity_matrix, request.param)

@pytest.fixture(params=VARIANTS_TO_TEST, scope='session')
def generate_network_f(request):
    return FuncWrapper(generate_network, request.param)

@pytest.fixture(params=VARIANTS_TO_TEST, scope='session')
def read_mgf_f(request):
    return FuncWrapper(read_mgf, request.param)

@pytest.fixture(params=VARIANTS_TO_TEST, scope='session')
def read_msp_f(request):
    return FuncWrapper(read_msp, request.param)

@pytest.fixture(params=VARIANTS_TO_TEST, scope='session')
def filter_data_f(request):
    return FuncWrapper(filter_data, request.param)

@pytest.fixture(params=VARIANTS_TO_TEST, scope='session')
def filter_data_multi_f(request):
    return FuncWrapper(filter_data_multi, request.param)

@pytest.fixture(params=VARIANTS_TO_TEST, scope='session')
def cosine_score_f(request):
    return FuncWrapper(cosine_score, request.param)

@pytest.fixture(params=VARIANTS_TO_TEST, scope='session')
def compare_spectra_f(request):
    return FuncWrapper(compare_spectra, request.param)

@pytest.fixture(params=VARIANTS_TO_TEST, scope='session')
def query_f(request):
    return FuncWrapper(query, request.param)
