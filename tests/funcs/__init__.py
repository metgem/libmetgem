import pytest
import warnings
import itertools

from libmetgem import IS_CYTHONIZED
from libmetgem.cosine import compute_similarity_matrix
from libmetgem.network import generate_network
from libmetgem.mgf import read as read_mgf
from libmetgem.msp import read as read_msp
from libmetgem.filter import filter_data, filter_data_multi
from libmetgem.cosine import (cosine_score,
                              entropy_score,
                              weighted_entropy_score,
                              compare_spectra)
from libmetgem.database import query
from libmetgem.neighbors import kneighbors_graph_from_similarity_matrix

    
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
        
        
class ScoreFuncGenerator:
    def __init__(self, variant):
        self.variant = variant
        
    def __call__(self, scoring, *args, **kwargs):
        match scoring:
            case 'cosine':
                func = cosine_score
            case 'entropy':
                func = entropy_score
            case 'weighted_entropy':
                func = weighted_entropy_score
        return FuncWrapper(func, self.variant)
        
    def __str__(self):
        return "<{} ({})>".format(self.__class__.__name__, self.variant)
        
    def __repr__(self):
        return self.__str__()
        
@pytest.fixture(scope='session')
def kneighbors_graph_from_similarity_matrix_f(variant):
    return FuncWrapper(kneighbors_graph_from_similarity_matrix, variant)
  
@pytest.fixture(scope='session')
def compute_similarity_matrix_f(variant):
    return FuncWrapper(compute_similarity_matrix, variant)

@pytest.fixture(scope='session')
def generate_network_f(variant):
    return FuncWrapper(generate_network, variant)

@pytest.fixture(scope='session')
def read_mgf_f(variant):
    return FuncWrapper(read_mgf, variant)

@pytest.fixture(scope='session')
def read_msp_f(variant):
    return FuncWrapper(read_msp, variant)

@pytest.fixture(scope='session')
def filter_data_f(variant):
    return FuncWrapper(filter_data, variant)

@pytest.fixture(scope='session')
def filter_data_multi_f(variant):
    return FuncWrapper(filter_data_multi, variant)

@pytest.fixture(scope='session')
def score_f_gen(variant):
    return ScoreFuncGenerator(variant)

@pytest.fixture(scope='session')
def compare_spectra_f(variant):
    return FuncWrapper(compare_spectra, variant)

@pytest.fixture(scope='session')
def query_f(variant):
    return FuncWrapper(query, variant)
