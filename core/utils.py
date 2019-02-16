import numpy as np
from collections import namedtuple

UniformDistribution = namedtuple('UniformDistribution', ['low', 'high'])
NormalDistribution = namedtuple('NormalDistribution', ['mean', 'stddev'])


def typed_arr(arr, dtype):
    return arr.astype(dtype)


def floats_arr(arr):
    return typed_arr(arr, np.float32)

def floats_uniform(low, high, n):
    return floats_arr(np.random.uniform(low, high, (n)))

def floats_normal(mean, stddev, n):
    return floats_arr(np.random.normal(mean, stddev, [n]))


def ints_arr(arr):
    return typed_arr(arr, np.int32)

def ints_uniform(low, high, n):
    return ints_arr(np.random.randint(low, high, (n)))

def ints_normal(mean, stddev, n):
    return ints_arr(np.random.normal(mean, stddev, [n]))


"""
Samplers: sometimes closures can be more concise than classes.
Are they as clear? These oughtta be, I hope...
"""

def list_sampler(l, dtype=np.float32):
    def ls(n):
        return typed_arr(np.array(l[:n]), dtype)
    return ls


def identical_sampler(v):
    return lambda n: floats_arr(np.ones((n,)) * v)

def uniform_sampler(low, high):
    return lambda n: floats_uniform(low, high, n)

def normal_sampler(mean, stddev):
    return lambda n: floats_normal(mean, stddev, n)


def identical_ints_sampler(v):
    return lambda n: ints_arr(np.ones((n,)) * v)

def uniform_ints_sampler(low, high):
    return lambda n: ints_uniform(low, high, n)

def normal_ints_sampler(mean, stddev):
    return lambda n: ints_normal(mean, stddev, n)
