import importlib
from functools import wraps
from types import FunctionType

__all__ = 'load_cython'

def load_cython(func):
    try:
        mod = importlib.import_module(func.__module__.replace('.', '._'))
        new_func = getattr(mod, func.__name__)
    except (ImportError, AttributeError) as e:
        print(e)
        return func
    else:
        if isinstance(func, FunctionType):
            @wraps(func)
            def inner(*args, **kwargs):
                return new_func(*args, **kwargs)       
            return inner
        else:
            return new_func