#
import numpy as np
from numba import njit
import torch

# 1. Jitted functions
@njit
def utility_jitted(a, m):
    
    c = (1-a)*m + 1e-200 # Add very small number to avoid log(0) -> NaN
    
    return np.log(c)

@njit
def marg_utility_jitted(a, m):

    c = (1-a)*m + 1e-200

    return np.divide(1, c)

@njit
def scrap_utility_jitted(b):

    return np.sqrt(b)

# 2. Not-jitted functions
def utility_notjitted(a, m):
    
    c = (1-a)*m + 1e-200 # Add very small number to avoid log(0) -> NaN
    
    return np.log(c)

def marg_utility_notjitted(a, m):

    c = (1-a)*m + 1e-200

    return np.divide(1, c)

def scrap_utility_notjitted(b):

    return np.sqrt(b)

# 3. Tensor-based functions
def utility_tensor(a, m):

    c = (1-a)*m + 1e-200

    return torch.log(c)

def marg_utility_tensor(a, m):

    c = (1-a)*m + 1e-200

    return torch.divide(1, c)

def scrap_utility_tensor(b):

    return torch.sqrt(b)