"""
    Test `libmetgem.human_readable_data`.
"""

import numpy as np
import pytest

from libmetgem import human_readable_data, MZ, INTENSITY
from funcs import filter_data_f

from data import random_spectrum


def test_human_readable_data_random(random_spectrum, filter_data_f):
    parent, data = random_spectrum

    filtered = filter_data_f(parent, data, 0, 17, 50, 6)
    data = human_readable_data(filtered)
       
    assert data.shape == filtered.shape
    
    max_ = filtered[:,INTENSITY].max() ** 2
    for i, row in enumerate(data):
        assert row[INTENSITY] == pytest.approx(filtered[i, INTENSITY] ** 2 / max_ *100)
        
    assert data[:,INTENSITY].max() == 100
    assert data[:INTENSITY].min() >= 0
    assert np.array_equal(data[:,MZ], filtered[:,MZ])
    assert np.array_equal(np.argsort(data[:,INTENSITY]),
                          np.argsort(filtered[:,INTENSITY]))
    
