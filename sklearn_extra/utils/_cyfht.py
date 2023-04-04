import numba

""" Implementation of the Fast Hadamard Transform.

https://en.wikipedia.org/wiki/Hadamard_transform

This module supplies a single dimensional and two-dimensional row-wise
implementation. Both are non-normalized, operate in-place and can only handle
the double/float64 type.

Inspired by a Python-C-API implementation at:

https://github.com/nbarbey/fht

"""


import numpy as np
from math import log2
 
def is_power_of_two(input_integer):
    """ Test if an integer is a power of two. """
    if input_integer == 1:
        return False
    return input_integer != 0 and ((input_integer & (input_integer - 1)) == 0)


def pure_python_fht(array_):
    """ Pure Python implementation for educational purposes. """
    bit = length = len(array_)
    for _ in range(int(np.log2(length))):
        bit >>= 1
        for i in range(length):
            if i & bit == 0:
                j = i | bit
                temp = array_[i]
                array_[i] += array_[j]
                array_[j] = temp - array_[j]


def fht(array_):
    """ Single dimensional FHT. """
    if not is_power_of_two(array_.shape[0]):
        raise ValueError('Length of input for fht must be a power of two')
    else:
        _fht(array_)


@numba.jit()
def _fht(array_):
    bit:int=0
    length:int=0
    i:int=0 
    j:int=0
    temp:float=0.0
    bit = length = array_.shape[0]
    for _ in range((log2(length))):
        bit >>= 1
        for i in range(length):
            if i & bit == 0:
                j = i | bit
                temp = array_[i]
                array_[i] += array_[j]
                array_[j] = temp - array_[j]

@numba.jit()
def fht2(array_):
    """ Two dimensional row-wise FHT. """
    if not is_power_of_two(array_.shape[1]):
        raise ValueError('Length of rows for fht2 must be a power of two')
    else:
        _fht2(array_)



def _fht2(array_):
    n = array_.shape[0]
    for x in range(n):
        _fht(array_[x])