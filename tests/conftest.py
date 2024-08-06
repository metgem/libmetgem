from libmetgem import IS_CYTHONIZED

SCORES_TO_TEST = {'cosine', 'entropy', 'weighted_entropy'}
NORM_TO_TEST = {'dot', 'sum'}

VARIANTS_TO_TEST = {'python'}
if IS_CYTHONIZED:
    VARIANTS_TO_TEST.add('cython')
    
def pytest_generate_tests(metafunc):
    if "scoring" in metafunc.fixturenames:
        metafunc.parametrize("scoring", SCORES_TO_TEST, scope='session')
    if "norm" in metafunc.fixturenames:
        metafunc.parametrize("norm", NORM_TO_TEST, scope='session')
    if "variant" in metafunc.fixturenames:
        metafunc.parametrize("variant", VARIANTS_TO_TEST, scope='session')
