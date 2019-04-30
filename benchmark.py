#!/usr/bin/env python3

"""
This file benchmarks two implementations of `christoffels`.

- christoffels_reference is the reference implementation in EinsteinPy, which uses SymPy.
- christoffels_symengine replaces the functions in SymPy with SymEngine.

Run Time:
- christoffels_reference: 0.20 seconds
- christoffels_symengine: 0.05 seconds

Note that this includes the overhead introduced by doing type conversion between SymPy
and SymEngine.
"""

import numpy as np
import sympy
import time
import symengine

def timeit(func):
    def function_timer(*args, **kwargs):
        """
        A nested function for timing other functions
        """
        start = time.time()
        value = func(*args, **kwargs)
        end = time.time()
        runtime = end - start
        msg = "The runtime for {func} took {time} seconds to complete"
        print(msg.format(func=func.__name__,
                         time=runtime))
        return value
    return function_timer

@timeit
def christoffels_reference(list2d, syms):
    dims = len(syms)
    christlist = (np.zeros(shape=(dims, dims, dims), dtype=int)).tolist()
    mat = sympy.Matrix(list2d)
    mat_inv = mat.inv()
    _counterlist = [i for i in range(dims ** 3)]
    for t in _counterlist:
        # i,j,k each goes from 0 to (dims-1)
        # hack for codeclimate. Could be done with 3 nested for loops
        k = t % dims
        j = (int(t / dims)) % (dims)
        i = (int(t / (dims ** 2))) % (dims)
        temp = 0
        for n in range(dims):
            temp += (mat_inv[i, n] / 2) * (
                sympy.diff(list2d[n][j], syms[k])
                + sympy.diff(list2d[n][k], syms[j])
                - sympy.diff(list2d[j][k], syms[n])
            )
        christlist[i][j][k] = temp
    return christlist

@timeit
def christoffels_symengine(list2d, syms):
    dims = len(syms)
    christlist = (np.zeros(shape=(dims, dims, dims), dtype=int)).tolist()
    mat = symengine.Matrix(list2d)
    mat_inv = mat.inv()
    _counterlist = [i for i in range(dims ** 3)]
    for t in _counterlist:
        # i,j,k each goes from 0 to (dims-1)
        # hack for codeclimate. Could be done with 3 nested for loops
        k = t % dims
        j = (int(t / dims)) % (dims)
        i = (int(t / (dims ** 2))) % (dims)
        temp = 0
        for n in range(dims):
            temp += (mat_inv[i, n] / 2) * (
                symengine.diff(list2d[n][j], syms[k])
                + symengine.diff(list2d[n][k], syms[j])
                - symengine.diff(list2d[j][k], syms[n])
            )
        # Converting back to SymPy type
        christlist[i][j][k] = sympy.S(temp)
    return christlist


def main():
    syms = sympy.symbols('r theta phi')
    # define the metric for 3d spherical coordinates
    metric = [[0 for i in range(3)] for i in range(3)]
    metric[0][0] = 1
    metric[1][1] = syms[0]**2
    metric[2][2] = (syms[0]**2)*(sympy.sin(syms[1])**2)
    ch_reference = christoffels_reference(metric, syms)
    ch_symengine = christoffels_symengine(metric, syms)

if __name__ == "__main__":
    main()
