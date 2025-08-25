#!/bin/env python3

def conditional_numba(function):
    """
        The conditional decorator will use numba if the package is installed 
        and otherwise revert to native python
    """
    try:
        from numba import jit
        return jit(cache=True,nopython=True,parallel=False)(function)
    except ModuleNotFoundError:
        print("Warning: {function.__name__}: numba not installed. For speedup please install numpy: pip install numba")
        return function
    
def conditional_jitclass(origclass):
    """
        The conditional decorator will use numba if the package is installed 
        and otherwise revert to native python
    """
    try:
        from numba.experimental import jitclass
        return jitclass()(origclass)
    except ModuleNotFoundError:
        print("Warning: {function.__name__}: numba not installed. For speedup please install numpy: pip install numba")
        return origclass