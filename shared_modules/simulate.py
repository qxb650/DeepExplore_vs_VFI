# Global modules
import torch
import numpy as np
from consav import linear_interp

# Local modules
import simprep
import state_funcs
import utility

def simulate_model(par, seed_dict, a_func, **kwargs):

    """
    Function that simulates model environment

    Args:

        par:                Parameter namespace

        seed_dict:          Dict of seeds for initial states and shocks

        a_func:             Functional (policy) that takes states and returns action
                            For tensors:        Pytorch Policy nn
                            For arrays:         RegularGridInterpolater

    Kwargs:

        iteration:          Iteration number (used for seed_dicts)

        eval_iteration:     Number of evaluation iteration (used for seed_dict)

        eval:               Boolean for whether it is a evaluation simulation (used for seed_dict)

        device:             If model is simulated using tensors, device must be specified

        tensor_bool:        Boolean for whether to simulate on tensors 

    Returns:

        obj:                For np.array:       Average lifetime utility
                            For torch.tensor:   Negative average lifetime utility (loss)

        state:              For np.array:       T x N x st (3)
                            For torch.tensor:   T x N x (st + 1) (4)

        actions:            T x N np.array or torch.tensor
    """

    # 1. Extract and check kwargs
    iteration = kwargs.get('iteration', None)
    eval_iteration = kwargs.get('eval_iteration', None)
    eval = kwargs.get('eval', False)
    device = kwargs.get('device', None)
    tensor_bool = kwargs.get('tensor_bool', False)

    if tensor_bool:
        assert device

        st = par.st + 1

    else:
        st = par.st

    # 2. Draw initial states and shocks
    # 2.1 Set seeds
    if eval:
        int_states_seed = seed_dict['seed_initial_states']
        shocks_seed = seed_dict['seed_shocks']

        N = par.N_eval

    else:
        int_states_seed = seed_dict['seed_initial_states'][eval_iteration][iteration]
        shocks_seed = seed_dict['seed_shocks'][eval_iteration][iteration]

        N = par.N_training

    # 2.2 Simulate initial states and shocks
    int_states = simprep.simulate_initial_states(par, int_states_seed, tensor_bool=tensor_bool, eval=eval, device=device)
    shocks = simprep.simulate_shocks(par, shocks_seed, tensor_bool=tensor_bool, eval=eval, device=device)

    # 3.1 Allocate memory for states
    states = np.zeros((par.T, N, st))
    actions = np.zeros((par.T, N))

    # 3.2 Fill states and actions
    if tensor_bool:
        # 3.2.1 Convert to tensors
        states = torch.tensor(states, dtype=par.dtype, device=device)
        actions = torch.tensor(actions, dtype=par.dtype, device=device)

        # 3.2.2 Initial states
        states[0] = int_states

        # 3a.4 Call simulation
        obj, states, actions = simulate_model_tensor(par, a_func, states, actions, shocks, device)

    else:
        # 3b.1 Initial states
        states[0] = int_states

        # 3b.2 Check that sol is filled and change dtype
        sol_policy = a_func.policy.astype(par.dtype)
        assert np.any(sol_policy != 0)
       
        # 3b.3 Call simulation
        obj, states, actions = simulate_model_array(par, sol_policy, states, actions, shocks)
    
    return obj, states, actions

def simulate_model_tensor(par, nn, states, actions, shocks, device):

    """
    
    Args:

        par:                Parameter namespace

        nn:                 Function for computing action

        states:             T x N x (st + 1) (4) torch.tensor, only with initial states, other will be filled out

        actions:            T x N empty torch.tensor to be filled out

        shocks:             (T - 1) x N torch.tensor

        device:             Device must be set

    Returns:

        loss:               Tensor element, negative average lifetime utility

        states:             T x N x (st + 1) (4) torch.tensor filled out from simulation

        actions:            T x N torch.tensor filled out from simulation

    """

    # 1. Memory
    T, N, n_st = states.shape
    utils = torch.zeros((T, N), dtype=par.dtype, device=device)

    # 2. Simulate all periods
    for t in range(par.T):

        # 2.1 Get state for period t
        if t > 0:
            st = new_st
        
        else:
            st = states[0] # Initial states

        if par.stateclamp:
            st = st.clamp(par.statebounds[0], par.statebounds[1]).clone()
        
        # 2.2 Extract states and shocks
        m = st[:, 0]
        R = st[:, 1]
        p = st[:, 2]

        if t < par.T - 1: # No state transition in last period
            xi_R = shocks[0][t] # First element of tuple, shock for the period t
            xi_p = shocks[1][t]

        # 2.3 Activate neural network
        a = nn(st).clamp(par.clamp_a[0], par.clamp_a[1]).clone().view(-1) # Always clamp actions, view is torch-command for reshape
        
        # 2.4 Store actions
        actions[t] = a

        # 2.5 Compute utils, savings and extrac
        utils[t] = (par.beta**t)*utility.utility_tensor(a, m)

        b = state_funcs.post_decision_state_notjitted(a, m)

        # 2.6 State transition
        if t < par.T - 1:
            mplusone, Rplusone, pplusone = state_funcs.state_transition_notjitted(par, b, R, p, xi_R, xi_p)

            # 2.6.1 Memory for gradient-friendly new states
            new_st = torch.zeros((N, n_st), dtype=par.dtype, device=device)

            # 2.6.2 Fill out
            new_st[:, 0] = mplusone
            new_st[:, 1] = Rplusone
            new_st[:, 2] = pplusone
            new_st[:, 3] = t+1

            # 2.6.3 Store new states in state tensor
            states[t+1] = new_st
        
        # 2.7 Scrap utility
        else:
            utils[t] = utils[t] + (par.beta**t)*utility.scrap_utility_tensor(b)

    # 3. Compute loss (average lifetime utility)
    loss = -torch.sum(utils)/N

    return loss, states, actions

def simulate_model_array(par, sol_policy, states, actions, shocks):

    """
    
    Args:

        par:                Parameter namespace

        interpolater:       Function for computing action

        states:             T x N x st (3) np.array, only with initial states, other will be filled out

        actions:            T x N np.array to be filled out

        shocks:             (T - 1) x N np.array
    
    Returns:

        obj                 (N, ) np.array, lifetime utility for each household

        states              T x N x st (3) np.array filled out by simulation

        actions             T x N np.array filled out by simulation

    """

    # 1. Memory for utils
    utils = np.zeros(actions.shape)

    # 2. Simulate all periods
    for t in range(par.T):

        # 2.1 Get states
        st = states[t]

        # 2.2 Extract states and shocks
        m = st[:, 0]
        R = st[:, 1]
        p = st[:, 2]

        if t < par.T - 1: # No state transition in last period
            xi_R = shocks[0][t] # First element of tuple, shock for the period t
            xi_p = shocks[1][t]

        # 2.3 Interpolate action (fills actions[t] up)
        linear_interp.interp_3d_vec(par.m, par.R, par.p, sol_policy[t], m, R, p, actions[t])

        a = actions[t]

        actions[t] = np.clip(a, 0.0, 1.0)

        # 2.4 Compute utils, savings
        utils[t] = (par.beta**t)*utility.utility_notjitted(a, m)

        b = state_funcs.post_decision_state_notjitted(a, m)

        # 2.5 State transition
        if t < par.T - 1:
            mplusone, Rplusone, pplusone = state_funcs.state_transition_notjitted(par, b, R, p, xi_R, xi_p)
            
            states[t+1, :, 0] = mplusone
            states[t+1, :, 1] = Rplusone
            states[t+1, :, 2] = pplusone

        # 2.6 Scrap utility
        else:
            utils[t] = utils[t] + (par.beta**t)*utility.scrap_utility_notjitted(b)
        
    # 3. Compute objective for each household
    obj = np.sum(utils, axis=0)

    return obj, states, actions