"""
    Read spectra from MSP files.
"""

from ._loader import load_cython

from typing import Tuple, Generator
import io
import numpy as np
import os


__all__ = ('read')

def to_float(string: str) -> float:
    try:
        return float(string.strip())
    except ValueError:
        return 0.

def read_data(line: str, f: io.IOBase, num_peaks: int) -> Generator[Tuple[float], None, None]:
    mz = intensity = ''
    icol = False  # whether we are in intensity column or not
    peaks_read = 0
    
    while True:
        if line == '\n':
            return
            
        if line[:5].upper() == 'NAME:':
            try:
                f.seek(f.tell()-len(line)-1, os.SEEK_SET)
            except io.UnsupportedOperation:
                pass
            return
        
        for char in line:
            if char in '()[]{}':  # Ignore brackets
                continue
            elif char in ' \t,;:\n':  # Delimiter
                if icol and mz and intensity:
                    yield float(mz), float(intensity)
                    peaks_read += 1
                    if peaks_read >= num_peaks:
                        return
                    mz = intensity = ''
                icol = not icol
            elif not icol:
                mz += char
            else:
                intensity += char
                
        line = f.readline()
        if not line:
            break
                
    if icol and mz and intensity:
        yield float(mz), float(intensity)
        

@load_cython
def read(filename: str, ignore_unknown: bool=False) -> Tuple[dict, np.ndarray]:
    """
        Read a file in `NIST Text Format of Individual Spectra (MSP)`_ and yields spectra.
    
    Args:
        filename: name of the file to read.
        ignore_unknown (optional): ignore unknown parameters, i.e. everything
            but PEPMASS, CHARGE, RTINSECONDS and MSLEVEL. This argument has no
            effect if `read` is not cythonized.
            
    Yields:
        params: Dictionary of spectrum's parameters
        data:   2D array containing spectrum data.
    
    .. _NIST Text Format of Individual Spectra (MSP):
        https://chemdata.nist.gov/mass-spc/ms-search/docs/Ver20Man_11.pdf
    
    """
    
    in_data = False
    num_peaks = 0
    params = {}
    data = []
    close = False

    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            
            if line == '':
                break
            elif line == '\n':
                continue
                
            if in_data:
                if num_peaks > 0:
                    data = list(read_data(line, f, num_peaks))
                    if len(data) > 0:
                        data = np.array(data, dtype=np.float32)
                    else:
                        data = np.empty((0, 2), dtype=np.float32)
                else:
                    data = np.empty((0, 2), dtype=np.float32)
                    
                in_data = False
                yield params, data
            else:                
                if line[:5].upper() == 'NAME:':
                    params = {'name': line[5:].strip(), 'synonyms': []}
                    num_peaks = 0
                    data = []
                elif ':' in line:
                    if line[:10].upper() == 'NUM PEAKS:':
                        num_peaks = int(line[10:].strip())
                        in_data = True
                    elif line[:3].upper() == 'MW:':
                        params['mw'] = int(line[3:].strip())
                    elif line[:8].upper() == 'SYNONYM:':
                        params['synonyms'].append(line[8:].strip())
                    elif line[:12].upper() == 'PRECURSORMZ:':
                        params['precursormz'] = to_float(line[12:])
                    elif line[:10].upper() == 'EXACTMASS:':
                        params['exactmass'] = to_float(line[10:])
                    elif line[:14].upper() == 'RETENTIONTIME:':
                        params['retentiontime'] = to_float(line[14:])
                    elif not ignore_unknown:
                        pos = line.find(':')
                        if pos > 0:
                            key = line[:pos].lower()
                            params[key] = line[pos+1:].strip()
    
    if params:
        yield params, np.empty((0, 2), dtype=np.float32)