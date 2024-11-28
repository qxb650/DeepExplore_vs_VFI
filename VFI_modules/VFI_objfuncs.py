# 
import numpy as np
from numba import njit
from consav import linear_interp

# Local modules
import utility
import state_funcs

@njit
def action_value_T(a, m):
    return utility.utility_jitted(a, m) + utility.scrap_utility_jitted(state_funcs.post_decision_state(a, m))

@njit
def obj_T(x, m):
    a = x[0]
    return action_value_T(a, m)

@njit
def action_value_t(a, par, sol, t, m, R, p):
        
        util = utility.utility_jitted(a, m)

        b = state_funcs.post_decision_state(a, m)

        # Interpolate expected value for next period given choice
        exp_postdec_value = linear_interp.interp_3d(par.b, par.R, par.p, sol.postdec_value[t], b, R, p)

        value = util + par.beta*exp_postdec_value

        return value

@njit
def obj_t(x, par, sol, t, m, R, p):
    a = x[0]
    return action_value_t(a, par, sol, t, m, R, p)