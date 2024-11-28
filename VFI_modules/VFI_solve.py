#
import quantecon as qe
import numpy as np
from numba import njit, prange

# Local modules
import VFI_objfuncs

@njit(parallel=True)
def solve_T(par, sol):
        
    # 1. Unpack policy and value grids
    po = sol.policy[par.T-1]
    va = sol.value[par.T-1]
            
    # 1. Loop through all possible state combinations (R and p dropped since they only affect post-decision value)
    for i_m in prange(len(par.m)):
        m = par.m[i_m]

        # 2.1 Set bounds numba-friendly
        bounds = np.zeros((1,2), dtype=par.dtype)
        bounds[0, 0] = 0.0
        bounds[0, 1] = 1.0

        # 2.2 Make Nelder-Mead- and numba-friendly guess
        guess = np.array([0.5], dtype=par.dtype)

        if i_m > 0:
            guess[0] = po[i_m - 1, i_m - 1, i_m - 1] # Make smart guess

        # 2.3 Optimize
        results = qe.optimize.nelder_mead(VFI_objfuncs.obj_T, guess, bounds=bounds, args=(m,))

        # 2.4 Store result in lookup table
        po[i_m] = results.x[0] # Acess first and only element
        va[i_m] = results.fun # Acess function value

@njit(parallel=True)
def solve_t(par, sol, t):

    # 1. Unpack policy and value grids
    po = sol.policy[t]
    va = sol.value[t]

    # 2. Loop through all possible state combinations
    for i_m in prange(len(par.m)):

        m = par.m[i_m]

        for i_R in prange(len(par.R)):

            R = par.R[i_R]

            for i_p in prange(len(par.p)):
                
                p = par.p[i_p]

                # 2.1 Set bounds numba-friendly
                bounds = np.zeros((1,2), dtype=par.dtype)
                bounds[0, 0] = 0.0
                bounds[0, 1] = 1.0

                # 2.2 Make Nelder-Mead- and numba-friendly guess
                guess = np.array([0.0], dtype=par.dtype)
                
                guess[0] = sol.policy[t+1, i_m, i_R, i_p] # Smart guess

                # 2.3 Optimize
                results = qe.optimize.nelder_mead(VFI_objfuncs.obj_t, guess, bounds=bounds, args=(par, sol, t, m, R, p))

                # 2.4 Store results in lookup table
                po[i_m, i_R, i_p] = results.x[0] # Acess first-and-only element
                va[i_m, i_R, i_p] = results.fun # Acess function value