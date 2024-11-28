#
import numpy as np
from numba import njit, prange
from consav import linear_interp

#
import state_funcs

@njit(parallel=True)
def exp_w_t(par, sol, t):

    # 1. Unpack post-decision grid
    postdec = sol.postdec_value[t]

    # 2. Loop over post-decision states
    for i_b in prange(len(par.b)):

        b = par.b[i_b]

        for i_R in prange(len(par.R)):

            R = par.R[i_R]

            for i_p in prange(len(par.p)):

                p = par.p[i_p]

                # 2.1 Loop over Gauss-Hermite weights and outcomes
                outer_sum = 0.0
                
                for i_xR in prange(len(par.xR)):
                    
                    xR = par.xR[i_xR]
                    wR = par.wR[i_xR]

                    inner_sum = 0.0

                    for i_xp in prange(len(par.xp)):

                        xp = par.xp[i_xp]
                        wp = par.wp[i_xp]
                        
                        # 2.1.1 Compute next periods states
                        mplusone, Rplusone, pplusone  = state_funcs.state_transition(par, b, R, p, xR, xp)

                        # 2.1.2 Interpolate value given states
                        vplusone = linear_interp.interp_3d(par.m, par.R, par.p, sol.value[t+1], mplusone, Rplusone, pplusone)

                        inner_sum += wp*vplusone

                    outer_sum += wR*inner_sum
                
                # 2.2 Store expectation as outer sum
                postdec[i_b, i_R, i_p] = outer_sum