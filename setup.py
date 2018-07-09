import sys
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

import numpy as np

if sys.platform == "win32":
    OPENMP_FLAGS = "/openmp"
    WIN32 = True
else:
    OPENMP_FLAGS = "-fopenmp"
    WIN32 = False

ext_modules = [
    Extension(
        "libmetgem.cosine",
        ["libmetgem/cosine.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=[OPENMP_FLAGS],
        extra_link_args=[OPENMP_FLAGS]
    ),
    Extension(
        "libmetgem.filter",
        ["libmetgem/filter.pyx"],
        include_dirs=[np.get_include()]
    ),
    Extension(
        "libmetgem.mgf",
        ["libmetgem/mgf.pyx"],
        include_dirs=[np.get_include()]
    ),
    Extension(
        "libmetgem.common",
        ["libmetgem/common.pyx"]
    )
]

setup(
    name='libmetgem',
    version='1.0',
    ext_modules=cythonize(ext_modules, compile_time_env={'WIN32': WIN32}),
)