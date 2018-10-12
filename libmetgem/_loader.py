import importlib
from functools import wraps

__all__ = 'load_cython'

def load_cython(func):
    try:
        mod = importlib.import_module(func.__module__.replace('.', '._'))
        new_func = getattr(mod, func.__name__)
    except (ImportError, AttributeError):
        return func
    else:
        @wraps(func)
        def inner(*args, **kwargs):
            return new_func(*args, **kwargs)       
        return inner