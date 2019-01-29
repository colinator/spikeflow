import numpy as np
from collections import namedtuple

UniformDistribution = namedtuple('UniformDistribution', ['low', 'high'])
NormalDistribution = namedtuple('NormalDistribution', ['mean', 'stddev'])


def floats_arr(arr):
    return arr.astype(np.float32)

def floats_uniform(low, high, n):
    return floats_arr(np.random.uniform(low, high, (n)))

def floats_normal(mean, stddev, n):
    return floats_arr(np.random.normal(mean, stddev, [n]))


"""
Samplers: sometimes closures can be more concise than classes.
Are they as clear? These oughtta be, I hope...
"""

def identical_sampler(v):
    return lambda n: floats_arr(np.ones((n,)) * v)

def uniform_sampler(low, high):
    return lambda n: floats_uniform(low, high, n)

def normal_sampler(mean, stddev):
    return lambda n: floats_normal(mean, stddev, n)
