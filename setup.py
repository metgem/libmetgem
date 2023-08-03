import os
import sys
import subprocess
from setuptools import setup, find_packages
from setuptools.extension import Extension, _have_cython

import numpy as np
import versioneer

HAS_CYTHON = _have_cython()
if HAS_CYTHON:
    from Cython.Build import cythonize

SRC_PATH = "libmetgem"

skip_build = os.environ.get('CYTHON_SKIP_BUILD', default=False)
if skip_build:
    HAS_CYTHON = False

if sys.platform == "win32":
    OPENMP_FLAGS = "/openmp"
elif sys.platform == "darwin":
    OPENMP_FLAGS = "-Xpreprocessor -fopenmp -lomp"
else:
    OPENMP_FLAGS = "-fopenmp"
    
# enable coverage by building with linetrace activated
linetrace = os.environ.get('CYTHON_LINETRACE', default=False)

directives = {'embedsignature': True, 'linetrace': linetrace}
macros = [('CYTHON_TRACE', '1'), ('CYTHON_TRACE_NOGIL', '1')] if linetrace else []
     
if HAS_CYTHON:
    ext_modules = [
            Extension(
                "libmetgem._common",
                [os.path.join(SRC_PATH, "_common.pyx")],
                include_dirs=[SRC_PATH, np.get_include()]
            ),
            Extension(
                "libmetgem._cosine",
                [os.path.join(SRC_PATH, "_cosine.pyx")],
                extra_compile_args=[OPENMP_FLAGS],
                extra_link_args=[OPENMP_FLAGS]
            ),
            Extension(
                "libmetgem._filter",
                [os.path.join(SRC_PATH, "_filter.pyx")]
            ),
            Extension(
                "libmetgem._mgf",
                [os.path.join(SRC_PATH, "_mgf.pyx")]
            ),
            Extension(
                "libmetgem._msp",
                [os.path.join(SRC_PATH, "_msp.pyx")]
            ),
            Extension(
                "libmetgem._database",
                [os.path.join(SRC_PATH, "_database.pyx"),
                 os.path.join(SRC_PATH, "sqlite", "sqlite3.c")],
                 include_dirs=[os.path.join(SRC_PATH, "sqlite")]
            ),
            Extension(
                "libmetgem._network",
                [os.path.join(SRC_PATH, "_network.pyx")]
            ),
            Extension(
                "libmetgem._neighbors",
                [os.path.join(SRC_PATH, "_neighbors.pyx")],
                extra_compile_args=[OPENMP_FLAGS],
                extra_link_args=[OPENMP_FLAGS]
            )
        ]
    
    for e in ext_modules:
        e.define_macros.extend(macros)
        
    ext_modules = cythonize(ext_modules,
                            compiler_directives=directives)
        
    install_requires = ["numpy", "scipy"]
    include_dirs = [np.get_include()]
else:
    ext_modules = []
    install_requires = ["numpy", "pyteomics", "scipy"]
    include_dirs = []


with open('libmetgem/_cython.py', 'w') as f:
    f.write(f"# THIS FILE IS GENERATED FROM LIBMETGEM SETUP.PY\n\nIS_CYTHONIZED = {bool(HAS_CYTHON)}\n")
    
setup(
    ext_modules=ext_modules,
    packages = find_packages(),
    include_dirs=include_dirs,
    install_requires=install_requires,
    zip_safe=False,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
