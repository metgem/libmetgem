from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

import numpy as np

ext_modules = [
    Extension(
        "libmetgem.cosine",
        ["libmetgem/cosine.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['/openmp'],
        extra_link_args=['/openmp']
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
    ext_modules=cythonize(ext_modules),
)