"""
    A library for molecular networking based on MS/MS spectra.
"""

from .common import MZ, INTENSITY
from ._cython import IS_CYTHONIZED
from .filter import square_root_data, normalize_data

import numpy as np

def human_readable_data(data: np.ndarray, square_intensities=True) -> np.ndarray:
    """
        Normalize an MS/MS spectrum to the spectrum's maximum intensity and
        revert `filter_data` square-root process.
        
    Args:
        data: A 2D array representing an MS/MS spectrum.
        square_intensities (bool): Whether to square intensities to revert
            `filter_data` square-root process.
        
    Returns:
        A copy of the array with intensities squared (optional) and normalised to maximum.
        
    See Also:
        filter_data
    """
    
    data = data.copy()
    if square_intensities:
        data[:, INTENSITY] = data[:, INTENSITY] ** 2
    data[:, INTENSITY] = data[:, INTENSITY] / data[:, INTENSITY].max() * 100
    return data

from . import _version
__version__ = _version.get_versions()['version']
