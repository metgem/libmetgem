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

MAJOR = 0
MINOR = 6
MICRO = 2
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
SRC_PATH = "libmetgem"

if "--skip-build" in sys.argv:
    HAS_CYTHON = False
    sys.argv.remove("--skip-build")


if sys.platform == "win32":
    OPENMP_FLAGS = "/openmp"
elif sys.platform == "darwin":
    OPENMP_FLAGS = "-Xpreprocessor -fopenmp -lomp"
else:
    OPENMP_FLAGS = "-fopenmp"
    
# enable coverage by building cython file by running build_ext with
# `--with-cython-coverage` enabled
linetrace = False
if '--with-cython-coverage' in sys.argv:
    linetrace = True
    sys.argv.remove('--with-cython-coverage')

directives = {'embedsignature': True, 'linetrace': False}
macros = []
if linetrace:
    directives['linetrace'] = True
    macros = [('CYTHON_TRACE', '1'), ('CYTHON_TRACE_NOGIL', '1')]
     
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
                            compile_time_env={'WIN32': sys.platform == 'win32'},
                            compiler_directives=directives)
        
    install_requires = ["numpy", "scipy"]
    setup_requires = ["cython>=0.28"] # Need Cython>=0.28 for read-only memoryview
    include_dirs = [np.get_include()]
else:
    ext_modules = []
    setup_requires = []
    install_requires = ["numpy", "pyteomics", "scipy"]
    include_dirs = []


with open('libmetgem/_cython.py', 'w') as f:
    f.write(f"# THIS FILE IS GENERATED FROM LIBMETGEM SETUP.PY\n\nIS_CYTHONIZED = {bool(HAS_CYTHON)}\n")

with open('README.rst', 'r') as f:
    LONG_DESCRIPTION = f.read()
    
setup(
    name="libmetgem",
    author = "Nicolas Elie",
    author_email = "nicolas.elie@cnrs.fr",
    url = "https://github.com/metgem/libmetgem",
    version = versioneer.get_version(),
    cmdclass = versioneer.get_cmdclass(),
    description = "Library for Molecular Networking calculations",
    long_description = LONG_DESCRIPTION,
    keywords = ["chemistry", "molecular networking", "mass spectrometry"],
    license = "GPLv3+",
    classifiers = ["Development Status :: 4 - Beta",
                   "Intended Audience :: Science/Research",
                   "Intended Audience :: Developers",
                   "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
                   "Operating System :: OS Independent",
                   "Topic :: Scientific/Engineering :: Bio-Informatics",
                   "Topic :: Scientific/Engineering :: Chemistry",
                   "Topic :: Software Development :: Libraries :: Python Modules",
                   "Programming Language :: Cython",
                   "Programming Language :: Python :: 3",
                   "Programming Language :: Python :: 3.2",
                   "Programming Language :: Python :: 3.3",
                   "Programming Language :: Python :: 3.4",
                   "Programming Language :: Python :: 3.5",
                   "Programming Language :: Python :: 3.6",
                   "Programming Language :: Python :: 3.7",
                   "Programming Language :: Python :: 3.8"],
    ext_modules=ext_modules,
    packages = find_packages(),
    include_dirs=include_dirs,
    setup_requires=setup_requires,
    install_requires=install_requires,
    zip_safe=False
)
