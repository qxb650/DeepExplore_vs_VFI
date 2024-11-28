# Imported modules
import time
import sys
import os
import numpy as np
import numba as nb
from scipy.interpolate import RegularGridInterpolator
from consav import quadrature, linear_interp
from EconModel import EconModelClass, jit
from numba import njit, prange
import matplotlib.pyplot as plt

# Acess paths to modules in folders
local_modules_path = [os.getcwd() + '/shared_modules', os.getcwd() + '/DeepExplore_modules', os.getcwd() + '/VFI_modules']
for path in local_modules_path:
    sys.path.append(path)

# Local modules
import utility
import state_funcs
import simulate
import euler_errors

# VFI-related modules
import VFI_solve
import VFI_postdec_solve

class VFI(EconModelClass):

    def settings(self):

        self.not_floats = ['m', 'b', 'R', 'p', 'T', 's', 'Ns', 'ghn', 'NEsuccess']

    def setup(self):

        par = self.par

        # 1. Preferences
        par.beta = 0.99
        
        # 2. Shocks
        par.muR = -0.00005
        par.mup = -0.00005
        par.sigmaR = 0.01 # When mu isn't specified, it is altered to be such that E[x] = 1, sigma is kept for the underlying normal distribution
        par.sigmap = 0.01
        par.rhoR = 0.9
        par.rhop = 0.9

        # 3. Number of states, shocks, periods, households in sim and number of Gauss-Hermite quadrature nodes + weights
        par.st = 3
        par.sh = 2
        par.T = 50
        par.N_eval = 50_000
        par.ghn = 10

        # 4. Simulation specifications
        par.m_int_min = 1.0
        par.m_int_max = 3.0
        par.R_int_min = 1.05
        par.R_int_max = 1.15
        par.p_int_min = 1.0
        par.p_int_max = 3.0        

        # 5. Solution specifications (grids)
        par.Ns_m = 200 # number of grid points for states
        par.Ns_R = 200
        par.Ns_p = 200
        par.Ns_b = 200

        par.m_min = 0.0
        par.m_max = 100.0
        par.R_min = 0.5
        par.R_max = 1.5
        par.p_min = 0.0
        par.p_max = 5.0

        # 6. Data type
        par.dtype = np.float64 # ConSav interpolation fails in float32

    def allocate(self):

        par = self.par
        sol = self.sol
        sim = self.sim

        # State and post-decision state grids
        par.m = np.linspace(par.m_min, par.m_max, par.Ns_m, dtype=par.dtype)
        par.b = np.linspace(par.m_min, par.m_max, par.Ns_b, dtype=par.dtype)
        par.R = np.linspace(par.R_min, par.R_max, par.Ns_R, dtype=par.dtype)
        par.p = np.linspace(par.p_min, par.p_max, par.Ns_p, dtype=par.dtype)

        # Shock grids
        par.xR, par.wR = quadrature.log_normal_gauss_hermite(n=par.ghn, sigma=par.sigmaR)
        par.xp, par.wp = quadrature.log_normal_gauss_hermite(n=par.ghn, sigma=par.sigmap)

        # Sol grids
        sol.policy = np.zeros((par.T, par.Ns_m, par.Ns_R, par.Ns_p), dtype=par.dtype)
        sol.value =  np.zeros((par.T, par.Ns_m, par.Ns_R, par.Ns_p), dtype=par.dtype)
        sol.postdec_value = np.zeros((par.T-1, par.Ns_b, par.Ns_R, par.Ns_p), dtype=par.dtype)
        sol.comp_time = np.zeros(par.T, dtype=par.dtype)

        # Sim grids
        sim.action = np.zeros((par.T, par.N_eval), dtype=par.dtype)
        sim.state = np.zeros((par.T, par.N_eval, par.st), dtype=par.dtype)
        sim.obj = np.zeros(par.N_eval, dtype=par.dtype)
        sim.euler_errors = np.zeros((par.T-1, par.N_eval), dtype=par.dtype)

    def solve(self):

        with jit(self) as model:

            par = model.par
            sol = model.sol

            for t in reversed(range(par.T)):

                timestamp = time.time()

                if t == par.T - 1:
                    VFI_solve.solve_T(par, sol)
                
                else:
                    VFI_postdec_solve.exp_w_t(par, sol, t)
                    VFI_solve.solve_t(par, sol, t)

                comp_time = time.time() - timestamp

                sol.comp_time[t] = comp_time
                
                print(f'Period {t} solved in {comp_time:.2f} seconds')

    def simulate(self, seed_dict):

        par = self.par
        sol = self.sol
        sim = self.sim

        test_dict = seed_dict['test']
        
        sim.obj, sim.state, sim.action = simulate.simulate_model(par, test_dict, sol, eval=True)
    
    def compute_euler_errors(self):
        
        with jit(self) as model:

            par = model.par
            sol = model.sol
            sim = model.sim

            for t in range(par.T-1):

                timestamp = time.time()
                euler_errors.euler_errors_sim(par, sol, sim, t)

                print(f'Period {t} solved in {time.time() - timestamp:.2f} seconds')
