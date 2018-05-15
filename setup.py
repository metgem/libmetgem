from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

import numpy as np

ext_modules = [
    Extension(
        "cosinelib.cosine",
        ["cosinelib/cosine.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['/openmp'],
        extra_link_args=['/openmp']
    ),
    Extension(
        "cosinelib.filter",
        ["cosinelib/filter.pyx"],
        include_dirs=[np.get_include()]
    ),
    Extension(
        "cosinelib.mgf",
        ["cosinelib/mgf.pyx"],
        include_dirs=[np.get_include()]
    ),
    Extension(
        "cosinelib.common",
        ["cosinelib/common.pyx"]
    )
]

setup(
    name='cosinelib',
    version='0.1',
    ext_modules=cythonize(ext_modules),
)