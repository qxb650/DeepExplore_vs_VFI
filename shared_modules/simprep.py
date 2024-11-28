# Global modules
import numpy as np
import torch

def simulate_initial_states(par, seed, **kwargs):

    """
    Function that creates the same uniformly-distributed initial_states for both np.array and torch.tensor given seed
    Random draws are done in np.array instead of torch.tensor to avoid inconsistency in random draws

    Args:
        par:            Parameter namespace, giving distribution parameters for random draws and size of np.array

        seed:           Seed to draw uniformly distributed random-variables from
    
    Kwargs:
        tensor_bool:    Convert random-draws from np.array to torch.tensor

        device:         If array is converted to tensor, device must be specified

        eval:           Boolean whether it is a evaluation (used in sizes)

    Returns:
        initial_states: For np.array:       N x st (3)
                        For torch.tensor    N x (st + 1) (4)
    """

    tensor_bool = kwargs.get('tensor_bool', False)
    device = kwargs.get('device', None)
    eval = kwargs.get('eval', False)

    if tensor_bool:
        assert device
    
    if eval:
        N = par.N_eval

    else:
        N = par.N_training
    
    # 1. Allocate memory
    initial_states = np.zeros((N, par.st))

    # 2. Set rng and fill with uniform dist
    rng = np.random.default_rng(seed=seed)
    
    # m
    initial_states[:, [0]] = rng.uniform(par.m_int_min, par.m_int_max, (N, 1))

    # R
    initial_states[:, [1]] = rng.uniform(par.R_int_min, par.R_int_max, (N, 1))

    # p
    initial_states[:, [2]] = rng.uniform(par.p_int_min, par.p_int_max, (N, 1))

    # 3. Convert to tensor
    if tensor_bool:
        initial_states = torch.tensor(initial_states, dtype=par.dtype, device=device)
        
        # 3.1 Add time dimension
        initial_states = torch.cat((initial_states, torch.zeros((N, 1), dtype=par.dtype, device=device)), dim=1)

    return initial_states
    
def simulate_shocks(par, seed, **kwargs):

    """
    Function that draws log-normally distributed shocks for both np.array and torch.tensor given seed
    Random draws are done in np.array instead of torch.tensor to avoid inconsistency in random draws

    Args:
        par:            Parameter namespace, giving distribution parameters for random draws and size of np.array

        seed:           Seed to draw log-normally distributed random-variables from
    
    Kwargs:
        tensor_bool:    Convert random-draws from np.array to torch.tensor

        device:         If array is converted to tensor, device must be specified

        eval:           Boolean for whether it is an evaluation (used in sizes)

    Returns:
        xi_R:           T-1 x N np.array or torch.tensor

        xi_p:           T-1 x N np.array or torch.tensor
    """

    tensor_bool = kwargs.get('tensor_bool', False)
    device = kwargs.get('device', None)
    eval = kwargs.get('eval', False)

    if tensor_bool:
        assert device
    
    if eval:
        N = par.N_eval

    else:
        N = par.N_training

    # 1. Set rng and size
    rng = np.random.default_rng(seed=seed)

    size = (par.T-1, N) # No shocks in first period

    # 2. Draw arrays of lognormal dist
    xi_R = rng.lognormal(par.muR, par.sigmaR, size)
    xi_p = rng.lognormal(par.mup, par.sigmap, size)

    # 3. Convert to tensor
    if tensor_bool:
        xi_R = torch.tensor(xi_R, dtype=par.dtype, device=device)
        xi_p = torch.tensor(xi_p, dtype=par.dtype, device=device)

    return xi_R, xi_p