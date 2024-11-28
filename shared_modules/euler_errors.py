# Global modules
import numpy as np
import numba
import torch
from consav import linear_interp, quadrature
from numba import njit, prange

# Local modules
import state_funcs
import utility

@njit(parallel=True)
def euler_errors_sim(par, sol, sim, t):

    # Numba friendly
    ee_sim = sim.euler_errors[t]

    # Loop over all households in period t
    for n in prange(par.N_eval):

        # Extract households states, action, post-decision state and MU
        m, R, p = sim.state[t, n]
        a = sim.action[t, n]
        b = state_funcs.post_decision_state(a, m)
        c = a*m
        MU = utility.marg_utility_jitted(a, m)

        # Compute expectation with Gauss-Hermite quadrature
        outer_sum = 0.0

        for i_xR in prange(len(par.xR)):
            
            xR = par.xR[i_xR]
            wR = par.wR[i_xR]

            inner_sum = 0.0

            for i_xp in prange(len(par.xp)):

                xp = par.xp[i_xp]
                wp = par.wp[i_xp]
                
                # State transition
                mplusone, Rplusone, pplusone  = state_funcs.state_transition(par, b, R, p, xR, xp)

                # Interpolate next-period action, compute next-period consumption
                aplusone = linear_interp.interp_3d(par.m, par.R, par.p, sol.policy[t+1], mplusone, Rplusone, pplusone)
                cplusone = aplusone*mplusone

                # Compute next-period MU
                MUplusone = utility.marg_utility_jitted(aplusone, mplusone)

                # Sum over inner GH nodes and weights
                inner_sum += wp*MUplusone
            
            # Sum over outer GH nodes and weights
            outer_sum += wR*inner_sum*xR

        # Store
        ee_sim[n] = np.abs(MU - par.beta*(R**par.rhoR)*outer_sum)

def euler_errors_nn_linalg(par, euler_errors_tensor, nn, batch, batch_size, state_batch, action_batch, t, device, GH_tuple):
    
    # state and action tensors: (N, 4) and (N, 1)

    # 1. Unpack GH-tuple
    xR = GH_tuple[0]
    wR = GH_tuple[1]
    xp = GH_tuple[2]
    wp = GH_tuple[3]

    # 2. Batch indices
    lower_batch_indx = batch*batch_size
    upper_batch_indx = (1 + batch)*batch_size

    ee_tensor = euler_errors_tensor[t, lower_batch_indx : upper_batch_indx]

    with torch.no_grad():

        # Loop over households in batch
        for n in range(batch_size):

            # Extract states, actions, savings, consumption and MU
            m, R, p = state_batch[n, :-1]
            a = action_batch[n]
            b = state_funcs.post_decision_state_notjitted(a, m)
            c = a*m
            MU = utility.marg_utility_tensor(a, m)

            # Change sizes of GH nodes and weights
            xR_sized = xR.view(1, -1)
            xp_sized = xp.view(-1, 1)

            wR_sized = wR.view(1, -1)
            wp_sized = wp.view(-1, 1)

            # State transition
            mplusone, Rplusone, pplusone = state_funcs.state_transition_notjitted(par, b, R, p, xR_sized, xp_sized)

            # Change sizes of next-period states to utilize linalg
            Rplusone = Rplusone.expand(len(xp), len(xR))
            pplusone = pplusone.expand(len(xp), len(xR))

            st_next = torch.stack([mplusone, Rplusone, pplusone, torch.full_like(mplusone, t+1)], dim=-1)

            aplusone = nn(st_next).clamp(par.clamp_a[0], par.clamp_a[1]).view(len(xp), len(xR))

            MUplusone = utility.marg_utility_tensor(aplusone, mplusone)

            # Compute inner GH sum
            inner_sum = (wp_sized*MUplusone).sum(dim=0)

            # Compute outer GH sum
            outer_sum = (wR_sized*inner_sum * xR_sized).sum()

            ee_tensor[n] = abs(MU - par.beta*(R**par.rhoR)*outer_sum)




# TO BE DELETED
def euler_errors_nn(par, euler_errors_array, nn, batch, batch_size, state_batch, action_batch, t):

    # state and action tensors: (N, 4) and (N, 1)

    q_xR, q_wR = torch.tensor(quadrature.log_normal_gauss_hermite(n=10, sigma=par.sigmaR), dtype=par.floattype)
    q_xp, q_wp = torch.tensor(quadrature.log_normal_gauss_hermite(n=10, sigma=par.sigmap), dtype=par.floattype)

    ee_array = euler_errors_array[t]

    with torch.no_grad():
        for n in range(batch_size):
            print(f'Computing Euler Error for household {n}')

            m, R, p = state_batch[n, :-1]

            a = action_batch[n]

            b = states.post_decision_state_notjitted(a, m)

            c = a*m

            MU = utility.marg_utility_tensor(a, m)

            outer_sum = 0.0

            for i_xp in range(len(q_xp)):

                xp = q_xp[i_xp]
                wp = q_xp[i_xp]

                inner_sum = 0.0

                for i_xR in range(len(q_xR)):

                    xR = q_xR[i_xR]
                    wR = q_wR[i_xR]

                    mplusone, Rplusone, pplusone = states.state_transition_notjitted(par, b, R, p, xR, xp)

                    st_next = torch.tensor([mplusone, Rplusone, pplusone, t+1], dtype=par.floattype)

                    aplusone = nn(st_next).clamp(par.clamp_a[0], par.clamp_a[1]).clone().view(-1)

                    MUplusone = utility.marg_utility_tensor(aplusone, mplusone)

                    inner_sum += wp*MUplusone

                outer_sum += wR*inner_sum*xR

            ee_array_indx = batch*batch_size + n

            ee_array[ee_array_indx] = abs(MU - par.beta*(R**par.rhoR)*outer_sum)

@njit(parallel=True)
def euler_errors_sol(par, sol, t):

    # Numba friendly
    ee_sol = sol.euler_errors_grid[t]

    # Loop over state_grid
    for i_m in prange(len(par.m)):

        m = par.m[i_m]

        for i_R in prange(len(par.R)):

            R = par.R[i_R]

            for i_p in prange(len(par.p)):

                p = par.p[i_p]

                # Extract a and get MU
                a = sol.policy[t, i_m, i_R, i_p]
                b = a*m
                MU = utility.marg_utility_jitted(a, m)

                # Compute expectation with Gauss-Hermite (GH) quadrature
                outer_sum = 0.0

                for i_xR in prange(len(par.xR)):
                    
                    xR = par.xR[i_xR]
                    wR = par.wR[i_xR]

                    inner_sum = 0.0

                    for i_xp in prange(len(par.xp)):

                        xp = par.xp[i_xp]
                        wp = par.wp[i_xp]

                        # State transition
                        mplusone, Rplusone, pplusone  = state_funcs.state_transition(par, b, R, p, xR, xp)

                        # Interpolate next-period action
                        aplusone = linear_interp.interp_3d(par.m, par.R, par.p, sol.policy[t+1], mplusone, Rplusone, pplusone)

                        # Compute next-period MU
                        MUplusone = utility.marg_utility_jitted(aplusone, mplusone)

                        # Sum over inner GH nodes and weights
                        inner_sum += wp*MUplusone
                    
                    # Sum over outer GH nodes and weights
                    outer_sum += wR*inner_sum*xR # Model environment should have xR in product
                
                # Store
                ee_sol[i_m, i_R, i_p] = abs(MU - par.beta*R*outer_sum)