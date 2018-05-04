from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np

ext_modules = [
    Extension(
        "cosine",
        ["cosine.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['/openmp'],
        extra_link_args=['/openmp']
    ),
    Extension(
        "filter",
        ["filter.pyx"],
        include_dirs=[np.get_include()]
    ),
    Extension(
        "mgf",
        ["mgf.pyx"],
        include_dirs=[np.get_include()]
    )
]

setup(
    ext_modules=cythonize(ext_modules),
)